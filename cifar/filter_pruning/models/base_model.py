from torch import nn 
from functools import reduce 

BLOCKS_LEVEL_SPLIT_CHAR = '.'


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_block(self, block_name):
        """
        get block from block name
        :param block_name: str - should be st like abc.def.ghk
        :param model: nn.Module - which model that block would be drawn from
        :return: nn.Module - required block
        """
        def _get_block(acc, elem):
            if elem.isdigit():
                layer = acc[int(elem)]
            else:
                layer = getattr(acc, elem)
            return layer

        return reduce(lambda acc, elem: _get_block(acc, elem), block_name.split(BLOCKS_LEVEL_SPLIT_CHAR), self)