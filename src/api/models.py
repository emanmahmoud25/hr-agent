"""Pydantic request / response models."""
from pydantic import BaseModel
from typing import Optional, Literal


class CVTextRequest(BaseModel):
    cv_text : str
    task    : Literal["classify","skills","interview","improve"] = "classify"
    max_tokens: Optional[int] = None


class SingleResult(BaseModel):
    filename   : str
    task       : str
    char_count : int
    result     : str
    elapsed_s  : float


class FullResult(BaseModel):
    filename   : str
    char_count : int
    classify   : str
    skills     : str
    interview  : str
    improve    : str
    elapsed_s  : float


class StatusResponse(BaseModel):
    device       : str
    active_lora  : Optional[str]
    switch_count : int
    call_count   : int
    adapters     : dict
