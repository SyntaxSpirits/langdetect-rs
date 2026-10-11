"""Print the code points for which str.isupper() is true in this Python, as JSON."""

import json
import sys
import unicodedata

json.dump(
    {
        "unidata_version": unicodedata.unidata_version,
        "upper": [c for c in range(0x110000) if chr(c).isupper()],
    },
    sys.stdout,
)
