from extensions import db

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Integer, default=0)
    file_type = db.Column(db.String(10), nullable=False)
    tags = db.relationship('Tag', secondary="content_tags", backref='contents')

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

content_tags = db.Table('content_tags',
                        db.Column('content_id', db.Integer, db.ForeignKey('content.id'), primary_key=True),
                        db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
                        )
