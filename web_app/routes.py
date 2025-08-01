import sys
import asyncio
from pathlib import Path
from collections import defaultdict
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import render_template, request, flash
import requests

from web_app import app, forms
from config import Config


def process_form_data(file_input: list, text_area_input: str) -> dict:
    predictions_len = None
    grouped_by_sent = None
    all_tags_summary = None

    if file_input or text_area_input:
        textarea_val = text_area_input if text_area_input else ""
        text_from_files = ""
        files = file_input
        for file in files:
            text_from_files += file.read().decode("utf-8") + "\n"
        
        data = {
            "text_list": [f"{textarea_val}\n{text_from_files}"],
            "sent_tokenizer": True
        }
        resp = requests.post(f"{Config.API_URL}/get_ner_prediction", json=data)
        data = resp.json()

        predictions_len = data["predictions_len"]
        grouped_by_sent = defaultdict(dict)

        all_tags_summary = defaultdict(list)
        for i in range(predictions_len):
            grouped_by_sent[f"Sent_{i+1}"] = {
                "raw_prediction": data["predictions"][i],
                "tokens": data["tokens"][i],
                "tags_count": data["tags_count"][i],
                "grouped_by_tags": data["grouped_by_tags"][i],
                "human_readable_grouped_by_tags": data["human_readable_grouped_by_tags"][i],
            }
            for key, val in data["human_readable_grouped_by_tags"][i].items():
                if isinstance(val, list):
                    all_tags_summary[key].extend(val)
                else:
                    all_tags_summary[key].append(val)
    
    return {
        "predictions_len": predictions_len,
        "grouped_by_sent": grouped_by_sent,
        "all_tags_summary": all_tags_summary
    }


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
async def home():
    form = forms.MainForm()
    if not app.config["TESTING"]:
        val_mode = 0
        form_validation = form.validate_on_submit()
    else:
        val_mode = 1
        form_validation = request.method == 'POST'

    predictions_len = None
    grouped_by_sent = None
    all_tags_summary = None
    if form_validation:
        file_input = form.file_input.data if form.file_input.data else []
        text_area_input = form.text_area.data if form.text_area.data else ""

        try:
            # res = process_form_data(file_input, text_area_input)
            res: dict = await asyncio.to_thread(process_form_data, file_input, text_area_input)
            predictions_len = res.get("predictions_len")
            grouped_by_sent = res.get("grouped_by_sent")
            all_tags_summary = res.get("all_tags_summary")
        except Exception as e:
            error_message = f"Error occured during processing (shieeeeet): {str(e)}"
            flash(error_message, "danger")
            app.logger.error(f"Error in process_form_data: {e}", exc_info=True)
        
        # return render_template("home.html", form=form)
    return render_template("home.html",
                            form=form,
                            predictions_len=predictions_len,
                            grouped_by_sent=grouped_by_sent,
                            all_tags_summary=all_tags_summary
                            )
