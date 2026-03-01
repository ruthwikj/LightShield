"""UUID-based tags for identifying prompt boundaries. Format and use are defined by the caller."""

from uuid import UUID, uuid4


class Tag:
    """A single session-unique identifier. Layer names, boundary format, and wrapping live elsewhere."""

    def __init__(self) -> None:
        self._uuid: UUID = uuid4()

    @property
    def short_id(self) -> str:
        """Return the first 8 characters of the UUID."""
        return self._uuid.hex[:8]

    def wrap(self, content: str) -> str:
        """Wrap a string with this tag at the start and end. Returns <LS_id>content</LS_id>."""
        return f"<LS_{self.short_id}>{content}</LS_{self.short_id}>"

    def __str__(self) -> str:
        """Return the tag as a string."""
        return f"<LS_{self.short_id}>"
