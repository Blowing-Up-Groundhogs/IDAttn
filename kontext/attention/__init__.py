from .attention_processor_APITA import FluxAPITAAttnProcessor
from .attention_processor_base import FluxAttnProcessor, FluxIPAdapterAttnProcessor


def get_attention_processor(attention_setting: str):
    if attention_setting == 'full':
        return FluxAttnProcessor()
    elif attention_setting == 'APITA':
        return FluxAPITAAttnProcessor()
    else:
        raise NotImplementedError(f"Attention setting {attention_setting} is not supported")
