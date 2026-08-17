"""Stage 04: replace image placeholders with captured alt text."""
import re
_PLACEHOLDER=re.compile(r"\[IMAGE_DESC_PLACEHOLDER_(\d+)\]")
class ImagePlaceholderResolutionStage:
    """Join preprocessor image captures to structural image sections."""
    name="resolve_images"
    def run(self,sections,image_alts):
        """Return resolved sections and remove empty image sections."""
        out=[]
        for sec in sections:
            if sec.section_type!="image_description": out.append(sec); continue
            def sub(m):
                i=int(m.group(1)); return image_alts[i].alt_text if i<len(image_alts) else ""
            sec.content=_PLACEHOLDER.sub(sub,sec.content).strip()
            if sec.content: out.append(sec)
        return out
