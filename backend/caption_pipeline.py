import asyncio
import time
import pandas as pd
from essential_generators import DocumentGenerator

async def pipe(image):
    sentence = DocumentGenerator()
    res = {"caption": sentence.sentence()}
    return res

