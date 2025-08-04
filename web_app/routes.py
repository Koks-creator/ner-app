import sys
import asyncio
import os
from datetime import datetime
from time import time
import uuid
import json
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import render_template, request, flash, jsonify, session, redirect, url_for
import requests

from web_app import app, forms
from config import Config


def process_form_data(file_input: list, text_area_input: str) -> Tuple[Dict, str]:
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
    res = {
        "predictions_len": predictions_len,
        "grouped_by_sent": grouped_by_sent,
        "all_tags_summary": all_tags_summary
    }
    result_id = str(uuid.uuid4())
    current_time = int(time())
    res_file = f"temp_{result_id}_{current_time}.json"
    with open(f"{Config.WEB_APP_TEMP_FOLDER}/{res_file}", "w") as f:
        json.dump(res, f)
    return res, res_file


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
async def home():
# def home():
    form = forms.MainForm()

    # if request.method == 'POST':
    #     form_valid = form.validate()
    #     app.logger.info(f"Form validation: {form_valid}")
    #     app.logger.info(f"Form errors: {form.errors}")
    #     app.logger.info(f"CSRF token valid: {form.csrf_token.validate(form)}")
    #     app.logger.info(f"Request form data: {request.form}")
    #     app.logger.info(f"Request files: {request.files}")

    if not app.config["TESTING"]:
        val_mode = 0
        form_validation = form.validate_on_submit()
    else:
        val_mode = 1
        form_validation = request.method == 'POST'

    predictions_len = None
    grouped_by_sent = None
    all_tags_summary = None
    res_file = None
    if form_validation:
        file_input = form.file_input.data if form.file_input.data else []
        text_area_input = form.text_area.data if form.text_area.data else ""

        try:
            # res, res_file = process_form_data(file_input, text_area_input)
            res, res_file = await asyncio.to_thread(process_form_data, file_input, text_area_input)
            predictions_len = res.get("predictions_len")
            grouped_by_sent = res.get("grouped_by_sent")
            all_tags_summary = res.get("all_tags_summary")

            session["res_file"] = res_file
        except Exception as e:
            error_message = f"Error occured during processing (shieeeeet): {str(e)}"
            flash(error_message, "danger")
            app.logger.error(f"Error in process_form_data: {e}", exc_info=True)
        
        # return render_template("home.html", form=form)
    # else:
    #     # Sprawdź, czy jest zapisane ID w sesji
    #     res_file = session.get("res_file")

    return render_template("home.html",
                            form=form,
                            predictions_len=predictions_len,
                            grouped_by_sent=grouped_by_sent,
                            all_tags_summary=all_tags_summary,
                            res_file=res_file
                            )

@app.route("/download-results/<res_file>", methods=["GET"])
def download_results(res_file: str):
    session_res_file = session.get("res_file")
    if not session_res_file or session_res_file != res_file:
        flash("Invalid result id", "warning")
        return redirect(url_for("home"))

    result_file_path = f"{Config.WEB_APP_TEMP_FOLDER}/{session_res_file}"

    if not os.path.exists(result_file_path):
        flash("Result expired or already downloaded", "warning")
        return redirect(url_for("home"))
    
    try:
        with open(result_file_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        
        response = jsonify(results_data)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"ner_results_{current_time}.json"
        
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "application/json"
        
        os.remove(result_file_path)
        
        return response
    
    except Exception as e:
        app.logger.error(f"Error downloading results: {e}", exc_info=True)
        flash("Error while downloading results", "error")
        return redirect(url_for("home"))

@app.route("/health", methods=["GET"])
def health_check():
    status = "Success"
    try:
        req = requests.get(Config.API_URL)
        if req.status_code != 200:
            app.logger.error(f"Could not connect to the api: {req.status_code=}")
            status = "Failed"
    except Exception:
         app.logger.error("Could not connect to the api", exc_info=True)
         status = "Failed"
    
    return {"ApiStatus": status}
