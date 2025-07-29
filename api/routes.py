import sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydantic import BaseModel
import asyncio
from fastapi import Body, HTTPException, status
from nltk.tokenize import sent_tokenize

from api import app, ner_predictor, logger, Config

class PredictionResponse(BaseModel):
    predictions: List[List[Tuple[str, str]]]
    tokens: List[List[str]]
    predictions_len: int
    tags_count: List[dict]
    grouped_by_tags: List[dict[str, list[str]]]
    human_readable_grouped_by_tags: List[dict[str, list[str]]]
    human_readable_tags_map: dict = Config.HUMAN_READABLE_TAGS_MAP

class HealthResponse(BaseModel):
    status: str

def count_tags(tagged_tokens) -> Counter:
    tags = [tag for _, tag in tagged_tokens]
    return Counter(tags)


def extract_named_entities(tagged_words: List[List[Tuple[str, str]]]) -> defaultdict[list]:
    entities = defaultdict(list)
    current_entity = ""
    current_tag = None

    for word, tag in tagged_words:
        # Przypadek tagu "O" (Outside)
        if tag == "O":
            if current_entity and current_tag:
                entities[current_tag].append(current_entity.strip())
                current_entity = ""
                current_tag = None
            continue
        
        # Obsługa tagów z prefiksami (B- lub I-)
        if "-" in tag:
            state, entity_type = tag.split("-")
            
            # Przypadek 1: Początek nowej encji (B-)
            if state == "B":
                # Zapisz poprzednią encję, jeśli istnieje
                if current_entity and current_tag:
                    entities[current_tag].append(current_entity.strip())
                # Rozpocznij nową encję
                current_entity = word
                current_tag = entity_type
            
            # Przypadek 2: Kontynuacja encji (I-)
            elif state == "I":
                # Standardowa kontynuacja - ten sam typ encji
                if current_tag == entity_type:
                    current_entity += " " + word
                # Edge case: Tag I- bez poprzedzającego B-
                else:
                    # Zapisz poprzednią encję, jeśli istnieje
                    if current_entity and current_tag:
                        entities[current_tag].append(current_entity.strip())
                    # Traktuj I- jako rozpoczęcie nowej encji
                    current_entity = word
                    current_tag = entity_type
        
        # Obsługa tagów bez prefiksów (jeśli takie istnieją)
        else:
            if current_entity and current_tag:
                entities[current_tag].append(current_entity.strip())
            current_entity = word
            current_tag = tag

    # Zapisz ostatnią encję, jeśli pozostała
    if current_entity and current_tag:
        entities[current_tag].append(current_entity.strip())
    
    return entities


def process_ner(text_list: List[str], sent_tokenizer: bool) -> dict:
    all_texts = []
    if sent_tokenizer:
        for t in text_list:
            sents = sent_tokenize(t)
            all_texts.extend(sents)
    else:
        all_texts = text_list
        
    predictions, tokens = ner_predictor.get_predictions(all_texts)
    tags_counters = []
    for pred in predictions:
        tags_counters.append(count_tags(pred))

    grouped_by_tags = [extract_named_entities(pred) for pred in predictions]

    human_readable_grouped_by_tags = []
    for grouped_by_tag in grouped_by_tags:
        human_readable_grouped_by_tags.append(
            {Config.HUMAN_READABLE_TAGS_MAP.get(key, key): value for key, value in grouped_by_tag.items()}
        )
    return {
        "predictions": predictions,
        "tokens": tokens,
        "predictions_len": len(predictions),
        "tags_count": tags_counters,
        "grouped_by_tags": grouped_by_tags,
        "human_readable_grouped_by_tags": human_readable_grouped_by_tags
    }

@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return HealthResponse(status="all green")

@app.post("/get_ner_prediction", response_model=PredictionResponse)
async def get_ner_prediction(text_list: list[str] = Body(...), sent_tokenizer: bool = Body(...)):
    try:
        res = await asyncio.to_thread(process_ner, text_list, sent_tokenizer)
        return PredictionResponse(**res)
    except HTTPException as http_ex:
        logger.error(f"HTTPException {http_ex}", exc_info=True)
        raise http_ex
    except Exception as e:
        logger.error(f"(status code 500) Internal server error {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error {e}")
