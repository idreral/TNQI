from collections import deque
from threading import Lock
import time
from sqlalchemy.sql import func
from models import Content, Tag

class TagEvaluator:
    def __init__(self, app, db, recalculation_threshold=15):
        self.app = app
        self.db = db
        self.recalculation_threshold = recalculation_threshold
        self.call_count = 0
        self.tag_scores = {}
        self.last_calculation_time = 0

        # Кэширование просмотров
        self.view_history = deque(maxlen=954)  # Последние 50 просмотренных ID
        self.history_lock = Lock()  # Для потокобезопасности

        self._recalculate_scores()

    def _get_tag_statistics(self):
        """Получает статистику по всем тегам"""
        tag_stats = {}

        # Получаем общий средний рейтинг
        overall_avg = self.db.session.query(func.avg(Content.rating)).scalar() or 0

        # Получаем статистику по каждому тегу
        tags = Tag.query.all()

        for tag in tags:
            # Контент с этим тегом
            contents_with_tag = tag.contents

            if not contents_with_tag:
                continue

            ratings = [c.rating for c in contents_with_tag]
            avg_rating = np.mean(ratings)
            count = len(ratings)

            # Вычисляем lift
            lift = avg_rating / overall_avg if overall_avg != 0 else 1.0

            # Вычисляем статистическую значимость (p-value)
            # Сравниваем с контентом без этого тега
            contents_without_tag = Content.query.filter(
                ~Content.tags.any(Tag.id == tag.id)
            ).all()

            ratings_without = [c.rating for c in contents_without_tag]

            if len(ratings) > 1 and len(ratings_without) > 1:
                t_stat, p_value = stats.ttest_ind(ratings, ratings_without)
            else:
                p_value = 1.0

            # Вычисляем coverage
            total_content = Content.query.count()
            coverage = count / total_content if total_content > 0 else 0

            # Вычисляем penalty за покрытие (оптимально 20-60%)
            if coverage < 0.05 or coverage > 0.8:
                coverage_penalty = 0.1
            elif coverage < 0.2 or coverage > 0.6:
                coverage_penalty = 0.5
            else:
                coverage_penalty = 1.0

            # Итоговый скор
            score = abs(lift - 1) * (-np.log10(max(p_value, 1e-10))) * coverage_penalty * np.sqrt(count)

            tag_stats[tag.id] = {
                'score': score,
                'lift': lift,
                'count': count,
                'avg_rating': avg_rating,
                'p_value': p_value,
                'coverage': coverage
            }

        return tag_stats

    def _recalculate_scores(self):
        """Пересчитывает оценки тегов"""
        with self.app.app_context():
            self.tag_scores = self._get_tag_statistics()
            self.last_calculation_time = time.time()
            print(f"[TagEvaluator] Scores recalculated at {time.strftime('%H:%M:%S')}")
            print(f"[TagEvaluator] Top 5 tags: {self.get_top_tags(5)}")

    def evaluate(self):
        """Вызывается при каждом использовании. Пересчитывает оценки каждые N вызовов"""
        self.call_count += 1

        if self.call_count >= self.recalculation_threshold:
            self._recalculate_scores()
            self.call_count = 0

        return self.tag_scores

    def get_top_tags(self, n=10, exclude_tags=None):
        """Возвращает топ-N тегов по качеству"""
        if exclude_tags is None:
            exclude_tags = []

        # Фильтруем и сортируем
        filtered_scores = {
            tag_id: stats for tag_id, stats in self.tag_scores.items()
            if tag_id not in exclude_tags and stats['count'] >= 2  # Минимум 2 использования
        }

        sorted_tags = sorted(filtered_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return sorted_tags[:n]

    def get_content_relevance_score(self, content, user_preferences=None):
        """Вычисляет релевантность контента на основе тегов"""
        if not content.tags:
            return 0

        total_score = 0
        tag_count = 0

        for tag in content.tags:
            if tag.id in self.tag_scores:
                tag_stats = self.tag_scores[tag.id]
                # Комбинируем скор тега с его lift
                tag_quality = tag_stats['score'] * (1 + tag_stats['lift'])
                total_score += tag_quality
                tag_count += 1

        # Среднее качество тегов контента
        avg_tag_quality = total_score / tag_count if tag_count > 0 else 0

        # Добавляем бонус за количество качественных тегов
        quantity_bonus = np.log1p(tag_count)  # Логарифмический бонус, чтобы не переоценивать много тегов

        final_score = avg_tag_quality * quantity_bonus

        return final_score

    def get_recommendations(self, base_content, n=8, exclude_ids=None):
        """Получает рекомендации на основе качества тегов"""
        if exclude_ids is None:
            exclude_ids = []

        exclude_ids.append(base_content.id)

        # Получаем теги базового контента
        base_tags = {tag.id for tag in base_content.tags}

        if not base_tags:
            # Если у контента нет тегов, возвращаем случайные
            return Content.query.filter(
                ~Content.id.in_(exclude_ids)
            ).order_by(func.random()).limit(n).all()

        # Получаем топ теги из тех, что есть у базового контента
        base_tag_scores = []
        for tag_id in base_tags:
            if tag_id in self.tag_scores:
                base_tag_scores.append((tag_id, self.tag_scores[tag_id]['score']))

        base_tag_scores.sort(key=lambda x: x[1], reverse=True)

        # Ищем контент с похожими качественными тегами
        candidates = Content.query.filter(
            ~Content.id.in_(exclude_ids)
        ).all()

        scored_candidates = []
        for candidate in candidates:
            candidate_score = self._calculate_similarity(base_tag_scores, candidate)
            if candidate_score > 0:
                scored_candidates.append((candidate, candidate_score))

        # Сортируем по схожести и берем топ-N
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Если недостаточно кандидатов, добавляем случайные
        recommendations = [c[0] for c in scored_candidates[:n]]

        if len(recommendations) < n:
            existing_ids = exclude_ids + [c.id for c in recommendations]
            additional = Content.query.filter(
                ~Content.id.in_(existing_ids)
            ).order_by(func.random()).limit(n - len(recommendations)).all()
            recommendations.extend(additional)

        # Перемешиваем для разнообразия
        np.random.shuffle(recommendations)

        return recommendations

    def _calculate_similarity(self, base_tag_scores, candidate):
        """Вычисляет схожесть контента на основе качественных тегов"""
        if not candidate.tags:
            return 0

        candidate_tags = {tag.id for tag in candidate.tags}

        total_similarity = 0
        matching_quality = 0

        for tag_id, score in base_tag_scores[:3]:  # Учитываем только топ-3 тега
            if tag_id in candidate_tags:
                total_similarity += score
                matching_quality += 1

        # Штраф за несоответствие
        extra_tags_penalty = len(candidate_tags - {tag_id for tag_id, _ in base_tag_scores}) * 0.1

        # Бонус за точное соответствие
        precision_bonus = matching_quality / len(base_tag_scores) if base_tag_scores else 0

        final_similarity = (total_similarity * (1 + precision_bonus)) / (1 + extra_tags_penalty)

        return final_similarity
    def add_to_history(self, content_id):
        """Добавляет контент в историю просмотров"""
        with self.history_lock:
            if content_id not in self.view_history:
                self.view_history.append(content_id)

    def get_recently_viewed(self, n=None):
        """Возвращает список недавно просмотренных ID"""
        with self.history_lock:
            if n is None:
                return list(self.view_history)
            return list(self.view_history)[-n:]

    def clear_history(self):
        """Очищает историю просмотров"""
        with self.history_lock:
            self.view_history.clear()

    def get_recommendations(self, base_content, n=8, exclude_ids=None, diversity_penalty=True):
        """Получает рекомендации с учетом истории просмотров"""
        if exclude_ids is None:
            exclude_ids = []

        # Добавляем базовый контент и историю в исключения
        exclude_ids.append(base_content.id)

        with self.history_lock:
            exclude_ids.extend(self.view_history)

        # Убираем дубликаты
        exclude_ids = list(set(exclude_ids))

        # Получаем теги базового контента
        base_tags = {tag.id for tag in base_content.tags}

        if not base_tags:
            # Если у контента нет тегов, возвращаем случайные (с учетом истории)
            available = Content.query.filter(
                ~Content.id.in_(exclude_ids)
            ).order_by(func.random()).limit(n).all()

            # Если не хватает, убираем старые из истории
            if len(available) < n:
                with self.history_lock:
                    # Оставляем только последние 5 в истории
                    recent_5 = list(self.view_history)[-5:] if len(self.view_history) > 5 else []
                    exclude_ids_reduced = [base_content.id] + recent_5

                available = Content.query.filter(
                    ~Content.id.in_(exclude_ids_reduced)
                ).order_by(func.random()).limit(n).all()

            return available

        # Ищем контент с похожими качественными тегами (исключая просмотренные)
        base_tag_scores = []
        for tag_id in base_tags:
            if tag_id in self.tag_scores:
                base_tag_scores.append((tag_id, self.tag_scores[tag_id]['score']))

        base_tag_scores.sort(key=lambda x: x[1], reverse=True)

        candidates = Content.query.filter(
            ~Content.id.in_(exclude_ids)
        ).all()

        # Если кандидатов мало, уменьшаем историю
        if len(candidates) < n:
            with self.history_lock:
                # Используем только последние 5 просмотров
                recent_5 = list(self.view_history)[-5:] if len(self.view_history) > 5 else []
                reduced_exclude = [base_content.id] + recent_5

            candidates = Content.query.filter(
                ~Content.id.in_(reduced_exclude)
            ).all()

        scored_candidates = []
        already_selected = []  # Для diversity penalty

        for candidate in candidates:
            candidate_score = self._calculate_similarity(base_tag_scores, candidate)
            if candidate_score > 0:
                # Добавляем diversity penalty, если включено
                if diversity_penalty and already_selected:
                    diversity_score = self._calculate_diversity_penalty(candidate, already_selected)
                    candidate_score *= diversity_score

                scored_candidates.append((candidate, candidate_score))

        # Сортируем по финальному скору
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Выбираем топ-N, но с учетом разнообразия
        recommendations = []
        for candidate, score in scored_candidates:
            if len(recommendations) >= n:
                break
            recommendations.append(candidate)
            if diversity_penalty:
                already_selected.append(candidate)

        # Если недостаточно, добавляем случайные (с учетом истории)
        if len(recommendations) < n:
            existing_ids = exclude_ids + [c.id for c in recommendations]
            additional = Content.query.filter(
                ~Content.id.in_(existing_ids)
            ).order_by(func.random()).limit(n - len(recommendations)).all()

            # Если все еще не хватает, берем из старых просмотров
            if len(recommendations) + len(additional) < n:
                with self.history_lock:
                    old_viewed = list(self.view_history)[:10]  # Берем старые просмотры
                additional_more = Content.query.filter(
                    Content.id.in_(old_viewed)
                ).order_by(func.random()).limit(
                    n - len(recommendations) - len(additional)
                ).all()
                additional.extend(additional_more)

            recommendations.extend(additional)

        # Перемешиваем для разнообразия
        np.random.shuffle(recommendations)

        return recommendations

    def _calculate_diversity_penalty(self, candidate, already_selected):
        """Вычисляет штраф за схожесть с уже выбранными рекомендациями"""
        if not already_selected:
            return 1.0

        max_similarity = 0
        candidate_tags = {tag.id for tag in candidate.tags}

        for selected in already_selected:
            selected_tags = {tag.id for tag in selected.tags}

            if candidate_tags and selected_tags:
                # Jaccard similarity
                intersection = len(candidate_tags & selected_tags)
                union = len(candidate_tags | selected_tags)
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)

        # Штраф обратно пропорционален максимальной схожести
        penalty = 1.0 / (1.0 + max_similarity * 2)
        return penalty

    def get_recommendations_with_exploration(self, base_content, n=8, exploration_rate=0.2):
        """Рекомендации с исследованием (explore vs exploit)"""
        import random

        if random.random() < exploration_rate:
            # Исследование: возвращаем случайный контент
            with self.history_lock:
                exclude_ids = [base_content.id] + list(self.view_history)

            recommendations = Content.query.filter(
                ~Content.id.in_(exclude_ids)
            ).order_by(func.random()).limit(n).all()

            return recommendations
        else:
            # Использование: возвращаем рекомендации на основе тегов
            return self.get_recommendations(base_content, n=n, diversity_penalty=True)
    def add_to_history(self, content_id):
        """Добавляет контент в историю просмотров"""
        with self.history_lock:
            if content_id not in self.view_history:
                self.view_history.append(content_id)

    def get_recently_viewed(self, n=None):
        """Возвращает список недавно просмотренных ID"""
        with self.history_lock:
            if n is None:
                return list(self.view_history)
            return list(self.view_history)[-n:]

    def clear_history(self):
        """Очищает историю просмотров"""
        with self.history_lock:
            self.view_history.clear()

    def get_recommendations(self, base_content, n=8, exclude_ids=None, diversity_penalty=True):
        """Получает рекомендации с учетом истории просмотров"""
        if exclude_ids is None:
            exclude_ids = []

        # Добавляем базовый контент и историю в исключения
        exclude_ids.append(base_content.id)

        with self.history_lock:
            exclude_ids.extend(self.view_history)

        # Убираем дубликаты
        exclude_ids = list(set(exclude_ids))

        # Получаем теги базового контента
        base_tags = {tag.id for tag in base_content.tags}

        if not base_tags:
            # Если у контента нет тегов, возвращаем случайные (с учетом истории)
            available = Content.query.filter(
                ~Content.id.in_(exclude_ids)
            ).order_by(func.random()).limit(n).all()

            # Если не хватает, убираем старые из истории
            if len(available) < n:
                with self.history_lock:
                    # Оставляем только последние 5 в истории
                    recent_5 = list(self.view_history)[-5:] if len(self.view_history) > 5 else []
                    exclude_ids_reduced = [base_content.id] + recent_5

                available = Content.query.filter(
                    ~Content.id.in_(exclude_ids_reduced)
                ).order_by(func.random()).limit(n).all()

            return available

        # Ищем контент с похожими качественными тегами (исключая просмотренные)
        base_tag_scores = []
        for tag_id in base_tags:
            if tag_id in self.tag_scores:
                base_tag_scores.append((tag_id, self.tag_scores[tag_id]['score']))

        base_tag_scores.sort(key=lambda x: x[1], reverse=True)

        candidates = Content.query.filter(
            ~Content.id.in_(exclude_ids)
        ).all()

        # Если кандидатов мало, уменьшаем историю
        if len(candidates) < n:
            with self.history_lock:
                # Используем только последние 5 просмотров
                recent_5 = list(self.view_history)[-5:] if len(self.view_history) > 5 else []
                reduced_exclude = [base_content.id] + recent_5

            candidates = Content.query.filter(
                ~Content.id.in_(reduced_exclude)
            ).all()

        scored_candidates = []
        already_selected = []  # Для diversity penalty

        for candidate in candidates:
            candidate_score = self._calculate_similarity(base_tag_scores, candidate)
            if candidate_score > 0:
                # Добавляем diversity penalty, если включено
                if diversity_penalty and already_selected:
                    diversity_score = self._calculate_diversity_penalty(candidate, already_selected)
                    candidate_score *= diversity_score

                scored_candidates.append((candidate, candidate_score))

        # Сортируем по финальному скору
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Выбираем топ-N, но с учетом разнообразия
        recommendations = []
        for candidate, score in scored_candidates:
            if len(recommendations) >= n:
                break
            recommendations.append(candidate)
            if diversity_penalty:
                already_selected.append(candidate)

        # Если недостаточно, добавляем случайные (с учетом истории)
        if len(recommendations) < n:
            existing_ids = exclude_ids + [c.id for c in recommendations]
            additional = Content.query.filter(
                ~Content.id.in_(existing_ids)
            ).order_by(func.random()).limit(n - len(recommendations)).all()

            # Если все еще не хватает, берем из старых просмотров
            if len(recommendations) + len(additional) < n:
                with self.history_lock:
                    old_viewed = list(self.view_history)[:10]  # Берем старые просмотры
                additional_more = Content.query.filter(
                    Content.id.in_(old_viewed)
                ).order_by(func.random()).limit(
                    n - len(recommendations) - len(additional)
                ).all()
                additional.extend(additional_more)

            recommendations.extend(additional)

        # Перемешиваем для разнообразия
        np.random.shuffle(recommendations)

        return recommendations

    def _calculate_diversity_penalty(self, candidate, already_selected):
        """Вычисляет штраф за схожесть с уже выбранными рекомендациями"""
        if not already_selected:
            return 1.0

        max_similarity = 0
        candidate_tags = {tag.id for tag in candidate.tags}

        for selected in already_selected:
            selected_tags = {tag.id for tag in selected.tags}

            if candidate_tags and selected_tags:
                # Jaccard similarity
                intersection = len(candidate_tags & selected_tags)
                union = len(candidate_tags | selected_tags)
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)

        # Штраф обратно пропорционален максимальной схожести
        penalty = 1.0 / (1.0 + max_similarity * 2)
        return penalty

    def get_recommendations_with_exploration(self, base_content, n=8, exploration_rate=0.2):
        """Рекомендации с исследованием (explore vs exploit)"""
        import random

        if random.random() < exploration_rate:
            # Исследование: возвращаем случайный контент
            with self.history_lock:
                exclude_ids = [base_content.id] + list(self.view_history)

            recommendations = Content.query.filter(
                ~Content.id.in_(exclude_ids)
            ).order_by(func.random()).limit(n).all()

            return recommendations
        else:
            # Использование: возвращаем рекомендации на основе тегов
            return self.get_recommendations(base_content, n=n, diversity_penalty=True)

