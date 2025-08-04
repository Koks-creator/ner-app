from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from flask_wtf.file import MultipleFileField, FileAllowed
from wtforms.validators import ValidationError, length

from config import Config

def max_files_count(max_count: int):
    def _max_files_count(form, field):
        # field.data to lista obiektów FileStorage
        if field.data:
            if len(field.data) > max_count:
                raise ValidationError(f"You can upload max {max_count} files.")
    return _max_files_count


def max_files_size(max_size_mb: int):
    max_size_bytes = max_size_mb * 1024 * 1024  # Konwersja MB na bajty
    
    def _validate(form, field):
        files = field.data
        if not files or len(files) == 0:
            return
        
        total_size = 0
        for f in files:
            if f:
                current_position = f.tell()
                
                f.seek(0, 2)
                file_size = f.tell()
                total_size += file_size
                
                f.seek(current_position)
        
        if total_size > max_size_bytes:
            raise ValidationError(
                f"Combined size of all files cannot be larger than {max_size_mb} MB. Current size:  {total_size / (1024 * 1024):.2f} MB."
            )
    
    return _validate

class MainForm(FlaskForm):
    text_area = TextAreaField("Paste your text here",
                              [length(max=10000)],
                              render_kw={"rows": 15},
                              )
    file_input = MultipleFileField("Upload files",
                                   validators=[FileAllowed(["txt"]),
                                               max_files_count(Config.WEB_APP_MAX_FILES),
                                               max_files_size(Config.WEB_APP_MAX_FILES_SIZE_MB)],
                                   )

    submit = SubmitField("Submit")
