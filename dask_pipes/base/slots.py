__all__ = ['BaseOperatorSlot']


class BaseOperatorSlot:
    def __init__(self, parent, slot):
        self.parent = parent
        self.slot = slot

    def __rshift__(self, other):
        other_slot = '*'
        if isinstance(other, BaseOperatorSlot):
            other_slot = other.slot
        return self.parent.__rshift__(other, self.slot, other_slot)

    def __lshift__(self, other):
        other_slot = '*'
        if isinstance(other, BaseOperatorSlot):
            other_slot = other.slot
        return self.parent.__lshift__(other, self.slot, other_slot)

    def __rrshift__(self, other):
        self.__lshift__(other)
        return self

    def __rlshift__(self, other):
        self.__rshift__(other)
        return self
