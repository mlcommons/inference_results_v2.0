#!/usr/bin/python3
"""
This script automatically generates the closed/NVIDIA/README.md from the MLPerf Influence Tutorials Confluence page.

To run the script, export your Confluence API credentials as CONFLUENCE_API_USER and CONFLUENCE_API_PASS first. Then run
the script from within the container.
"""

import os
import sys
sys.path.insert(0, os.getcwd())
os.environ["OUTSIDE_MLPINF_ENV"] = "1" # To allow us to execute this script outside of the docker

from dataclasses import dataclass
from code.common.constants import VERSION
from scripts.internal.docgen.confluence import G_CONFLUENCE_REQUESTER
from scripts.internal.docgen.markdown_converter import html_to_markdown
from typing import Final, Optional, List


@dataclass(frozen=True)
class DocumentationPage:
    """
    Describes a documentation page located from Confluence that needs to be converted
    """

    local_file_name: str
    """str: The filename to save the page under"""

    confluence_page_id: int
    """int: The page ID of the page on Confluence"""

    page_header: str
    """str: The string to insert into the top of the file."""

G_DOC_PAGES: Final[List[DocumentationPage]] = [
    DocumentationPage(
        "README.md",
        341529806,
        "\n".join([
            f"# MLPerf Inference {VERSION} NVIDIA-Optimized Implementations",
            "This is a repository of NVIDIA-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.",
            "This README is a quickstart tutorial on how to use our code as a public / external user.",
            "\n**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "README_Triton_CPU.md",
        916782376,
        "\n".join([
            f"# MLPerf Inference {VERSION} NVIDIA-Optimized Implementations of Triton Inference Server running on CPU",
            "This is a repository of NVIDIA-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.",
            "This README is a quickstart tutorial on how to use our code for Triton on CPU systems as a public / external user.",
            "It is recommended to also read README.md for general instructions",
            "\n**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/performance_tuning_guide.md",
        895258873,
        "\n".join([
            "# NVIDIA MLPerf Inference System Under Test (SUT) Performance Tuning Guide",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/commands.md",
        895261438,
        "\n".join([
            "# MAKE Targets and RUN_ARGS Documentation",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/FAQ.md",
        895261350,
        "\n".join([
            "# Common Issues FAQ",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/submission_guide.md",
        895260904,
        "\n".join([
            "# MLPerf Inference Submission Guide",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/calibration.md",
        895261380,
        "\n".join([
            "# MLPerf Inference Calibration and Quantization Details",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
    DocumentationPage(
        "documentation/heterogeneous_mig.md",
        895261385,
        "\n".join([
            "# MLPerf Inference Heterogeneous MIG Workloads",
            "**NOTE**: This document is autogenerated from internal documentation. If something is wrong or confusing, please contact NVIDIA.",
            "\n---\n"
        ])
    ),
]


def pull_documentation_page(docpage):
    """
    Pulls a Confluence documentation page and writes the body out to a local file as Markdown.

    Args:
        docpage (DocumentationPage):
            a DocumentationPage object containing the information for the page to pull
    """
    html = G_CONFLUENCE_REQUESTER.get_page_contents(docpage.confluence_page_id)
    md = docpage.page_header + "\n" + html_to_markdown(html)
    with open(docpage.local_file_name, 'w') as f:
        f.write(md)


if __name__ == "__main__":
    for docpage in G_DOC_PAGES:
        print(f"Pulling Confluence PageID={docpage.confluence_page_id} to {docpage.local_file_name}")
        pull_documentation_page(docpage)
