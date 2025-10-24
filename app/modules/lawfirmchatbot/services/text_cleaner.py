import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Keep \t, \n, \r but remove other control characters
NON_PRINTABLE = dict.fromkeys(i for i in range(32) if i not in (9, 10, 13))


def normalize_text(text: str) -> str:
    """
    Normalize text with proper Unicode handling and smart punctuation replacement.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text with cleaned punctuation and whitespace
    """
    if not text:
        return ""
    
    # Unicode normalize & keep punctuation
    text = unicodedata.normalize("NFKC", text)
    
    # Remove control characters (keep tab, newline, carriage return)
    text = text.translate(NON_PRINTABLE)
    
    # Replace common Word smart characters with ASCII equivalents
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",  # various dashes
        "\u2018": "'", "\u2019": "'",  # smart single quotes
        "\u201C": '"', "\u201D": '"',  # smart double quotes
        "\u2026": "...",  # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Collapse excessive whitespace but keep paragraphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def looks_garbled(text: str) -> bool:
    """
    Detect if text is garbled/unreadable binary garbage.
    
    Uses two criteria:
    1. Printable ratio: percentage of printable characters
    2. Alphabetic density: percentage of actual letters
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be garbled, False if readable
    """
    if not text:
        return True
    
    # Calculate printable ratio
    printable = sum(ch.isprintable() for ch in text)
    printable_ratio = printable / max(1, len(text))
    
    # Calculate alphabetic density
    alpha = sum(ch.isalpha() for ch in text)
    alpha_ratio = alpha / max(1, len(text))
    
    # "garbled" only if BOTH look bad
    is_garbled = printable_ratio < 0.90 and alpha_ratio < 0.40
    
    if is_garbled:
        logger.warning(
            f"[text-cleaner] Detected garbled text: "
            f"printable_ratio={printable_ratio:.2f}, alpha_ratio={alpha_ratio:.2f}"
        )
    
    return is_garbled


def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted document text to remove unreadable symbols,
    control characters, binary junk, and extra spaces.
    Preserves meaningful punctuation and structure.
    """

    if not text:
        return ""

    # Use improved normalization
    text = normalize_text(text)

    # Remove repeating junk characters (�, nulls, weird sequences)
    text = re.sub(r"[�•□■▪▫¤◆◇◈○●◊⬜⬛⬤■□▀▄█]+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def clean_and_validate_text(text: str, filename: str = "", mark_garbled: bool = True) -> tuple[str, dict]:
    """
    Enhanced cleaning with garbled detection and quality metadata.
    
    Args:
        text: Raw extracted text
        filename: Optional filename for logging context
        mark_garbled: Whether to detect and mark garbled text
        
    Returns:
        Tuple of (cleaned_text, metadata_dict)
        metadata_dict contains "quality": "ok" | "garbled" | "empty"
    """
    metadata = {"quality": "ok"}
    
    if not text:
        logger.warning(f"Empty text received for file: {filename}")
        metadata["quality"] = "empty"
        return "", metadata
    
    original_length = len(text)
    cleaned_text = clean_extracted_text(text)
    cleaned_length = len(cleaned_text)
    
    # Check if cleaned text looks garbled
    if mark_garbled and looks_garbled(cleaned_text):
        logger.warning(f"Garbled text detected for {filename}: text appears unreadable")
        metadata["quality"] = "garbled"
        # Still return the text, but mark it so caller can decide what to do
    
    # Log significant cleanup (indicates potential issues with extraction)
    if original_length > 0:
        cleanup_ratio = (original_length - cleaned_length) / original_length
        if cleanup_ratio > 0.5:  # More than 50% removed
            logger.warning(
                f"Significant text cleanup for {filename}: "
                f"removed {cleanup_ratio*100:.1f}% of content "
                f"({original_length} -> {cleaned_length} chars)"
            )
        metadata["cleanup_ratio"] = cleanup_ratio
    
    return cleaned_text, metadata
