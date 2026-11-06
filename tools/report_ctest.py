"""Expose failed CTest output as GitHub check annotations."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def escape(value):
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    path = Path(sys.argv[1])
    if not path.exists():
        return  # Compilation/setup failures may precede CTest.
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.find("failure") is not None:
            output = case.findtext("system-out") or case.findtext("failure") or "Failed"
            print(f"::error title={escape(case.attrib['name'])}::{escape(output)}")


if __name__ == "__main__":
    main()
