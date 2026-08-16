from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import numpy as np
from PIL import Image
from pathlib import Path
import os
import uuid
import cv2
import time
from collections import defaultdict
from scipy import stats
from collections import deque  # ← Добавьте эту строку
from threading import Lock      # ← И эту для потокобезопасности
import time                     # ← И эту, если еще не импортировали
from scipy import stats

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///content.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREVIEW_FOLDER'] = 'static/uploads/preview'
app.config['SECRET_KEY'] = 'secretkey'
db = SQLAlchemy(app)

ip = "127.0.0.1"

PREVIEW_SIZE = (200, 200)
GIF_DURATION = 5

# Models
class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Integer, default=0)
    file_type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    tags = db.relationship('Tag', secondary='content_tags', backref='contents')

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

content_tags = db.Table('content_tags',
    db.Column('content_id', db.Integer, db.ForeignKey('content.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

class TagEvaluator:
    def __init__(self, recalculation_threshold=15):
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
        overall_avg = db.session.query(func.avg(Content.rating)).scalar() or 0

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
        with app.app_context():
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




def create_image_preview(image_path, output_path):
    try:
        image = Image.open(image_path).convert("RGBA")
        image.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
        
        # Создаем новое изображение с прозрачным фоном
        preview = Image.new("RGBA", PREVIEW_SIZE, (0, 0, 0, 0))
        x_offset = (PREVIEW_SIZE[0] - image.width) // 2
        y_offset = (PREVIEW_SIZE[1] - image.height) // 2
        preview.paste(image, (x_offset, y_offset))
        
        preview.save(output_path, "PNG")
        print(f"Создано превью для {image_path}")
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {e}")

def create_video_preview(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Не удалось открыть видео")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Определяем коэффициент ускорения
        speed_factor = max(1, duration / GIF_DURATION)
        frame_interval = int(speed_factor)
        
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = Image.fromarray(frame)
                frame.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
                
                # Создаем кадр с прозрачным фоном
                preview_frame = Image.new("RGBA", PREVIEW_SIZE, (0, 0, 0, 0))
                x_offset = (PREVIEW_SIZE[0] - frame.width) // 2
                y_offset = (PREVIEW_SIZE[1] - frame.height) // 2
                preview_frame.paste(frame, (x_offset, y_offset))
                frames.append(preview_frame)
            count += 1
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=42,  # Устанавливаем фиксированное время кадра (1000/24 ≈ 42 мс)
                loop=0
            )
            print(f"Создано превью для {video_path}")
        cap.release()
    except Exception as e:
        print(f"Ошибка обработки видео {video_path}: {e}")

def build_filter_query(tag_ids, mode):
    """Возвращает базовый запрос Content, отфильтрованный по списку тегов с учётом режима."""
    query = Content.query
    if not tag_ids:
        return query
    if mode == 'strict':
        subquery = (
            db.session.query(Content.id, func.count(Tag.id).label("match_count"))
            .select_from(Content)
            .join(content_tags, Content.id == content_tags.c.content_id)
            .join(Tag, Tag.id == content_tags.c.tag_id)
            .filter(Tag.id.in_(tag_ids))
            .group_by(Content.id)
            .having(func.count(Tag.id) == len(tag_ids))
            .subquery()
        )
        query = query.join(subquery, Content.id == subquery.c.id)
    elif mode == 'soft':
        subquery = (
            db.session.query(Content.id)
            .select_from(Content)
            .join(content_tags, Content.id == content_tags.c.content_id)
            .filter(content_tags.c.tag_id.in_(tag_ids))
            .distinct()
            .subquery()
        )
        query = query.join(subquery, Content.id == subquery.c.id)
    elif mode == 'exclude':
        excluded_subquery = (
            db.session.query(Content.id)
            .join(content_tags, Content.id == content_tags.c.content_id)
            .filter(content_tags.c.tag_id.in_(tag_ids))
            .distinct()
            .subquery()
        )
        query = query.filter(Content.id.notin_(excluded_subquery))
    return query

@app.route('/search')
def index():
    selected_tags = request.args.getlist('tags', type=int)
    page = request.args.get('page', 1, type=int)
    per_page = 20
    mode = request.args.get('mode', 'strict')

    # Основной запрос с текущими выбранными тегами
    main_query = build_filter_query(selected_tags, mode)
    total = main_query.count()  # общее количество до пагинации

    # Пагинация
    pagination = main_query.order_by(Content.rating.desc()).paginate(page=page, per_page=per_page, error_out=False)
    contents = pagination.items

    # Считаем для каждого тега количество при добавлении
    all_tags = Tag.query.all()
    tag_counts = {}
    for tag in all_tags:
        if tag.id in selected_tags:
            # Тег уже выбран – количество не изменится
            tag_counts[tag.id] = total
        else:
            # Добавляем этот тег к текущему набору
            new_tags = selected_tags + [tag.id]
            new_query = build_filter_query(new_tags, mode)
            tag_counts[tag.id] = new_query.count()

    return render_template(
        'index.html',
        contents=contents,
        pagination=pagination,
        tags=all_tags,
        selected_tags=[str(t) for t in selected_tags],
        mode=mode,
        tag_counts=tag_counts
    )

@app.route('/rate/<int:content_id>/<string:change>', methods=["POST"])
def rate(content_id, change):
    tag_evaluator.evaluate()
    content = Content.query.get_or_404(content_id)
    if change == 'up':
        content.rating += 1
    elif change == 'down':
        content.rating -= 1
    db.session.commit()
    return jsonify({"success": True, "new_rating": content.rating})

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form['title']
        file = request.files['file']
        selected_tags = request.form.getlist('tags')
        
        if file:
            file_ext = file.filename.rsplit('.', 1)[-1]
            name_uuid = uuid.uuid4().hex
            unique_filename = f"{name_uuid}.{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            file_type = 'image' if file_ext.lower() in ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'] else 'video'

            new_content = Content(title=title, file_path=file_path, file_type=file_type)

            if file_type == 'image':
                output_file = os.path.join(app.config['PREVIEW_FOLDER'],f"{name_uuid}.png")
                create_image_preview(file_path, output_file)
            else:
                output_file = os.path.join(app.config['PREVIEW_FOLDER'],f"{name_uuid}.gif")
                create_video_preview(file_path, output_file)
            
            for tag_id in selected_tags:
                tag = Tag.query.get(tag_id)
                if tag:
                    new_content.tags.append(tag)
            
            db.session.add(new_content)
            db.session.commit()
            return redirect(url_for('index'))
    
    tags = Tag.query.order_by(Tag.name).all()
    return render_template('upload.html', tags=tags)

@app.route('/tags-manager', methods=['GET', 'POST'])
def manage_tags():
    if request.method == 'POST':
        tag_name = request.form['tag_name'].strip()
        if tag_name:
            existing_tag = Tag.query.filter_by(name=tag_name).first()
            if not existing_tag:
                new_tag = Tag(name=tag_name)
                db.session.add(new_tag)
                db.session.commit()
    tags = Tag.query.all()
    return render_template('tags.html', tags=tags)

@app.route('/edit_tags/<int:content_id>', methods=['POST'])
def edit_tags(content_id):
    content = Content.query.get_or_404(content_id)
    selected_tags = request.form.getlist('tags')

    # Обновляем теги
    content.tags = Tag.query.filter(Tag.id.in_(selected_tags)).all()
    
    db.session.commit()
    
    return {"success": True, "updated_tags": [tag.name for tag in content.tags]}

@app.route('/tags')
def get_tags():
    tags = Tag.query.order_by(Tag.name).all()
    tags_data = [{"id": tag.id, "name": tag.name} for tag in tags]
    
    content_id = request.args.get("content_id")
    content_tags = []
    
    if content_id:
        content = Content.query.get(content_id)
        if content:
            content_tags = [tag.id for tag in content.tags]
    
    return {"tags": tags_data, "content_tags": content_tags}

@app.route('/edit_tag/<int:tag_id>', methods=['POST'])
def edit_tag(tag_id):
    tag = Tag.query.get_or_404(tag_id)
    new_name = request.form.get('new_tag_name')
    if not new_name:
        flash('Tag name cannot be empty!', 'error')
    elif Tag.query.filter_by(name=new_name).first():
        flash('Tag name already exists!', 'error')
    else:
        tag.name = new_name
        db.session.commit()
        flash('Tag updated successfully!', 'success')
    return redirect(url_for('manage_tags'))

@app.route('/delete_tag/<int:tag_id>', methods=['POST'])
def delete_tag(tag_id):
    tag = Tag.query.get_or_404(tag_id)
    db.session.delete(tag)
    db.session.commit()
    return redirect(url_for('manage_tags'))

@app.route('/delete_content/<int:content_id>')
def delete_content(content_id):
    content = Content.query.get_or_404(content_id)
    if os.path.exists(content.file_path):
        os.remove(content.file_path)
    db.session.delete(content)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/')
def random_content():
    # Обновляем оценки тегов
    tag_evaluator.evaluate()

    # Получаем параметр seed
    seed_id = request.args.get('seed', type=int)

    if seed_id:
        main_content = Content.query.get(seed_id)
        if not main_content:
            main_content = Content.query.order_by(func.random()).first()
    else:
        main_content = Content.query.order_by(func.random()).first()

    # Добавляем текущий контент в историю просмотров
    tag_evaluator.add_to_history(main_content.id)

    # Получаем рекомендации с учетом истории и diversity penalty
    # Используем exploration для разнообразия (20% случайных)
    recommendations = tag_evaluator.get_recommendations_with_exploration(
        main_content,
        n=6,
        exploration_rate=0.2
    )

    # Добавляем рекомендации в историю
    for rec in recommendations:
        tag_evaluator.add_to_history(rec.id)

    tags = Tag.query.all()

    # Получаем топ теги для отображения
    top_tags = tag_evaluator.get_top_tags(10)
    top_tags_with_names = []
    for tag_id, stats in top_tags:
        tag = Tag.query.get(tag_id)
        if tag:
            top_tags_with_names.append({
                'tag': tag,
                'stats': stats
                })

    return render_template(
        'random.html',
        main_content=main_content,
        recommendations=recommendations,
        tags=tags,
        top_tags=top_tags_with_names,
        evaluator_stats={
            'last_update': tag_evaluator.last_calculation_time,
            'calls_until_update': tag_evaluator.recalculation_threshold - tag_evaluator.call_count,
            'history_size': len(tag_evaluator.get_recently_viewed())
        }
    )

@app.route('/tag-stats')
def tag_stats():
    # Принудительно обновляем статистику
    tag_evaluator._recalculate_scores()

    stats_list = []
    for tag_id, stats in tag_evaluator.tag_scores.items():
        tag = Tag.query.get(tag_id)
        if tag:
            stats_list.append({
                'tag': tag,
                'stats': stats
            })

    # Сортируем по скору
    stats_list.sort(key=lambda x: x['stats']['score'], reverse=True)

    return render_template(
        'tag_stats.html',
        stats_list=stats_list,
        total_content=Content.query.count()
    )

@app.route('/edit_title/<int:content_id>', methods=['POST'])
def edit_title(content_id):
    content = Content.query.get_or_404(content_id)
    new_title = request.form.get('new_title', '').strip()
    if new_title:
        content.title = new_title
        db.session.commit()
        return jsonify({"success": True, "new_title": content.title})
    return jsonify({"success": False}), 400

@app.route('/history')
def view_history():
    """Показать историю просмотров (для отладки)"""
    history_ids = tag_evaluator.get_recently_viewed()
    history_content = []
    for content_id in history_ids:
        content = Content.query.get(content_id)
        if content:
            history_content.append(content)

    return render_template(
        'history.html',
        history=history_content,
        history_count=len(history_content)
    )

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Очистить историю просмотров"""
    tag_evaluator.clear_history()
    flash('View history cleared!', 'success')
    return redirect(url_for('index'))

# ... (предыдущий код, включая импорты и модели) ...

# Глобальное состояние для сортировки
sort_state = None

@app.route('/sort')
def sort_page():
    global sort_state
    # Инициализация состояния, если его нет или был сброс
    if sort_state is None or 'content_ids' not in sort_state:
        all_contents = Content.query.order_by(Content.rating.asc()).all()
        if len(all_contents) < 2:
            return "Not enough content to sort (need at least 2).", 400
        content_ids = [c.id for c in all_contents]
        processed = [False] * len(content_ids)
        sort_state = {
            'content_ids': content_ids,
            'processed': processed,
            'current_index': 0,        # индекс первого необработанного элемента
            'completed': 0,
            'history': []              # для Undo
        }

    # Поиск текущей пары для сравнения
    ids = sort_state['content_ids']
    processed = sort_state['processed']
    current = sort_state['current_index']

    # Пропускаем уже обработанные элементы в начале (если current указывает на обработанный)
    while current < len(ids) and processed[current]:
        current += 1
    if current >= len(ids) - 1:
        # Все элементы обработаны или остался один
        return render_template('sort.html',
                               state=sort_state,
                               finished=True,
                               left_item=None,
                               right_item=None)

    # Ищем следующий необработанный элемент после current
    right_idx = current + 1
    while right_idx < len(ids) and processed[right_idx]:
        right_idx += 1
    if right_idx >= len(ids):
        # Нет пары для сравнения — сортировка завершена
        return render_template('sort.html',
                               state=sort_state,
                               finished=True,
                               left_item=None,
                               right_item=None)

    left_item = Content.query.get(ids[current])
    right_item = Content.query.get(ids[right_idx])
    return render_template('sort.html',
                           state=sort_state,
                           finished=False,
                           left_item=left_item,
                           right_item=right_item)

@app.route('/sort/choose', methods=['POST'])
def sort_choose():
    global sort_state
    if sort_state is None:
        return redirect(url_for('sort_page'))

    winner_side = request.form.get('winner')  # 'left' или 'right'
    if winner_side not in ('left', 'right'):
        return redirect(url_for('sort_page'))

    ids = sort_state['content_ids']
    processed = sort_state['processed']
    current = sort_state['current_index']

    # Находим реальные индексы left и right (пропуская обработанные)
    left_idx = current
    while left_idx < len(ids) and processed[left_idx]:
        left_idx += 1
    right_idx = left_idx + 1
    while right_idx < len(ids) and processed[right_idx]:
        right_idx += 1

    if right_idx >= len(ids):
        # Нечего сравнивать – завершено
        return redirect(url_for('sort_page'))

    left_id = ids[left_idx]
    right_id = ids[right_idx]
    left = Content.query.get(left_id)
    right = Content.query.get(right_id)

    # Создаём снапшот для Undo
    snapshot = {
        'content_ids': list(ids),
        'processed': list(processed),
        'current_index': sort_state['current_index'],
        'completed': sort_state['completed'],
        'changed_id': None,
        'old_rating': None
    }

    if winner_side == 'left':
        # Левый победил
        if left.rating <= right.rating:
            snapshot['changed_id'] = left.id
            snapshot['old_rating'] = left.rating
            left.rating = right.rating + 1
            db.session.commit()
        # Пересортировка всех элементов по рейтингу (возрастание)
        items = []
        for i, cid in enumerate(ids):
            rating = Content.query.get(cid).rating
            items.append((cid, rating, processed[i]))
        items.sort(key=lambda x: x[1])  # по возрастанию рейтинга
        new_ids = [item[0] for item in items]
        new_processed = [item[2] for item in items]

        # Находим новую позицию left
        new_left_pos = new_ids.index(left.id)
        # Ищем следующий необработанный элемент после new_left_pos
        next_unprocessed = None
        for i in range(new_left_pos + 1, len(new_ids)):
            if not new_processed[i]:
                next_unprocessed = i
                break

        if next_unprocessed is not None:
            # Продолжаем сравнивать left со следующим необработанным
            sort_state['current_index'] = new_left_pos
        else:
            # Left стал последним необработанным – помечаем его обработанным
            new_processed[new_left_pos] = True
            # Ищем следующий необработанный для новой итерации
            next_start = None
            for i in range(new_left_pos + 1, len(new_ids)):
                if not new_processed[i]:
                    next_start = i
                    break
            if next_start is None:
                sort_state['current_index'] = len(new_ids)  # завершение
            else:
                sort_state['current_index'] = next_start

        sort_state['content_ids'] = new_ids
        sort_state['processed'] = new_processed
    else:  # winner_side == 'right'
        # Правый победил – повышаем его рейтинг, если нужно
        if right.rating <= left.rating:
            snapshot['changed_id'] = right.id
            snapshot['old_rating'] = right.rating
            right.rating = left.rating + 1
            db.session.commit()
        # Левый фиксируется как обработанный
        processed[left_idx] = True
        # Пересортировка всех элементов по рейтингу (возрастание) с учётом новых рейтингов
        items = []
        for i, cid in enumerate(ids):
            rating = Content.query.get(cid).rating
            items.append((cid, rating, processed[i]))
        items.sort(key=lambda x: x[1])
        new_ids = [item[0] for item in items]
        new_processed = [item[2] for item in items]

        # Находим позицию победителя (right)
        new_right_pos = new_ids.index(right.id)
        # Ищем следующий необработанный элемент после new_right_pos
        next_unprocessed = None
        for i in range(new_right_pos + 1, len(new_ids)):
            if not new_processed[i]:
                next_unprocessed = i
                break

        if next_unprocessed is not None:
            # Продолжаем сравнивать right со следующим необработанным
            sort_state['current_index'] = new_right_pos
        else:
            # Right стал последним необработанным – помечаем его обработанным
            new_processed[new_right_pos] = True
            # Ищем следующий необработанный для новой итерации
            next_start = None
            for i in range(new_right_pos + 1, len(new_ids)):
                if not new_processed[i]:
                    next_start = i
                    break
            if next_start is None:
                sort_state['current_index'] = len(new_ids)
            else:
                sort_state['current_index'] = next_start

        sort_state['content_ids'] = new_ids
        sort_state['processed'] = new_processed

    sort_state['completed'] += 1
    sort_state['history'].append(snapshot)

    return redirect(url_for('sort_page'))

@app.route('/sort/undo', methods=['POST'])
def sort_undo():
    global sort_state
    if sort_state is None or not sort_state.get('history'):
        return redirect(url_for('sort_page'))

    snapshot = sort_state['history'].pop()
    sort_state['content_ids'] = snapshot['content_ids']
    sort_state['processed'] = snapshot['processed']
    sort_state['current_index'] = snapshot['current_index']
    sort_state['completed'] = snapshot['completed']

    # Откат изменения рейтинга, если было
    if snapshot['changed_id'] is not None:
        content = Content.query.get(snapshot['changed_id'])
        if content:
            content.rating = snapshot['old_rating']
            db.session.commit()

    return redirect(url_for('sort_page'))

@app.route('/sort/reset', methods=['POST'])
def sort_reset():
    global sort_state
    sort_state = None
    return redirect(url_for('sort_page'))

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    print("Bipki")
    # Папка, где лежат загруженные видео (может быть внутри static или отдельно)
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    filepath = os.path.join(upload_folder, filename)
    
    if not os.path.exists(filepath):
        abort(404)
    
    # send_file автоматически распознаёт MIME-тип и поддерживает Range
    return send_file(filepath, conditional=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Инициализация оценщика тегов
    tag_evaluator = TagEvaluator(recalculation_threshold=1000)
    app.run(debug=True, host=ip)
