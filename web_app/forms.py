from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from flask_wtf.file import MultipleFileField, FileAllowed
from wtforms.validators import DataRequired, ValidationError

from config import Config

def max_files_count(max_count: int):
    def _max_files_count(form, field):
        # field.data to lista obiektów FileStorage
        if field.data:
            if len(field.data) > max_count:
                raise ValidationError(f"You can upload max {max_count} files.")
    return _max_files_count


class MainForm(FlaskForm):
    text_area = TextAreaField("Paste your text here",
                              render_kw={"rows": 15}
                              )
    file_input = MultipleFileField("Upload files",
                                   validators=[FileAllowed(["txt"]), max_files_count(Config.WEB_APP_MAX_FILES)],
                                   )

    submit = SubmitField("Submit")