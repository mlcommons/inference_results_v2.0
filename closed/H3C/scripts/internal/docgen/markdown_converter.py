import os
import sys
sys.path.insert(0, os.getcwd())

import html
import re

from bs4 import BeautifulSoup
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, unique
from scripts.internal.docgen.confluence import G_CONFLUENCE_REQUESTER, ConfluenceInternalTagArgs
from typing import Final, Dict, Optional, Set, List


G_METATAG_PATTERN: Final[re.Pattern] = re.compile(r"!!\[(.*?)\]")
"""re.Pattern: Pattern that matches the special tag syntax used in MLPerf Influence documentation on Confluence to
denote things like internal/action items

group(1): ", "-separated list of tag parameters
"""


@dataclass(frozen=True)
class MDSpecialCharacter:
    """
    Special character to serve as an intermediary symbol as some MarkDown symbols interfere interfere when parsing out
    the !!internal tag.
    """
    intermediary_symbol: str
    """str: Symbol to use as the intermediary representation"""

    raw_symbol: str
    """str: Symbol to translate into when generating the final string"""


@unique
class MDSpecialCharacters(Enum):
    """
    Special characters that may cause issues when parsing HTML -> !!internal tags -> MarkDown.
    """
    SquareBracketOpen = MDSpecialCharacter("<sqbracket_open>", "[")
    SquareBracketClose = MDSpecialCharacter("<sqbracket_close>", "]")


class ListItemIndex:
    """
    Tracks a list item's index.
    """

    def __init__(self, index_str_format, initial_value=1):
        self.value = initial_value
        self.initial_value = initial_value
        self.index_str_format = index_str_format

    def __call__(self):
        index_str = self.index_str_format.format(self.value)
        self.value += 1
        return index_str

    def new(self):
        """
        Used to create a unique instance of a ListItemIndex

        Returns:
            ListItemIndex: a new ListItemIndex with the same initial_value and the new value reset to initial.
        """
        return ListItemIndex(self.index_str_format, initial_value=self.initial_value)


@dataclass(frozen=True)
class HTMLMarkdownToken:
    """
    Describes a HTML token and its equivalent formatter in Markdown
    """
    html_token: str
    """str: The string between the '<', '>' denoting the HTML formatter token"""

    md_token_format: Optional[str]
    """str: The string denoting a format string to insert the HTML token body into to achieve the equivalent format in
    Markdown. If None, denotes that this is a parent HTML node, like <ul> or <ol>."""

    inc_indent: bool = False
    """bool: Whether or not this token should increment the indent depth for its children. If True, global indent depth
    is incremented, and will be decremented when this format token scope is exitted. Default: False"""

    list_item_index: Optional[ListItemIndex] = None
    """ListItemIndex: If not None, used as the current string prefix to denote a list item. Default: None"""


G_KNOWN_HTML_TOKENS: Dict[str, HTMLMarkdownToken] = {
    tok.html_token: tok
    for tok in [
        HTMLMarkdownToken("p", "{}\n\n"),
        HTMLMarkdownToken("ac:inline-comment-marker", "{}"),
        HTMLMarkdownToken("span", "{}"),
        HTMLMarkdownToken("li", "{}\n"),
        HTMLMarkdownToken("h1", "## {}\n\n"),
        HTMLMarkdownToken("h2", "### {}\n\n"),
        HTMLMarkdownToken("h3", "### {}\n\n"),
        HTMLMarkdownToken("ul", "{}", inc_indent=True, list_item_index=ListItemIndex("-")),
        HTMLMarkdownToken("ol", "{}", inc_indent=True, list_item_index=ListItemIndex("{}.")),
        HTMLMarkdownToken("noformat", "```\n{}\n```\n"),
        HTMLMarkdownToken("code", "`{}`"),
        HTMLMarkdownToken("strong", "**{}**"),
        HTMLMarkdownToken("em", "*{}*"),
        HTMLMarkdownToken("i", "*{}*"),
    ]
}


G_LIST_TAGS: Final[Set[str]] = {"ul", "ol"}


def html_to_markdown(html_str):
    """
    Traverses through the HTML tree via depth-first search and generates the lines for an equivalent Markdown document.

    Args:
        An HTML string to parse

    Returns:
        A string representing a Markdown document
    """
    # Do some minor cleanup to move spaces out of <bold> blocks
    for tag in ["code", "strong", "em", "i"]:
        html_str = html_str.replace(f" </{tag}>", f"</{tag}> ")

    # Arbitrary special symbol as a placeholder for a space. Leading spaces in <span> blocks are formatted as &nbsp and
    # aren't parsed correctly. Use a special symbol rather than a space, since BS4 will discard any leading spaces.
    html_str = html_str.replace("&nbsp;", "!_!")

    # Use Beautiful soup to tokenize the text as an HTML tree
    tokens = BeautifulSoup(html_str, "html.parser")

    def get_token_as_md(token, indent_level=0, list_prefixer=None):
        """
        Recursive traversal step to pass through each of the HTML tree's nodes. Note that it is fine that this is a
        recursive method, as the HTML tree depth will be very shallow (height usually at most 5) due to the nature of
        how it is structured as a Confluence document.
        """
        # Base case: Not an HTML formatted node. Return as raw string
        if token.name is None:
            return token.string
        elif token.name == "ac:plain-text-body":
            # Handle "plain-text" / unformatted text differently.
            return token.string
        elif token.name == "br":
            return "\n\n"
        elif token.name == "pre":
            # <pre> blocks need to be html-unescaped
            return html.unescape(token.string)
        elif token.name == "a":
            # Handle hyperlinks differently, since href is a BeautifulSoup node variable
            href = token.get("href")
            sq_bracket_open = MDSpecialCharacters.SquareBracketOpen.value.intermediary_symbol
            sq_bracket_close = MDSpecialCharacters.SquareBracketClose.value.intermediary_symbol
            return f"{sq_bracket_open}{token.string}{sq_bracket_close}({href})"
        elif token.name == "ac:link":
            children = list(token.children)
            if len(children) != 1:
                raise RuntimeError(f"<ac:link> HTML node should contain exactly 1 link. Got: {token}")
            child = children[0]
            if child.name == "ri:user":
                return "<ConfluenceUser>"
            elif child.name == "ri:page":
                return f"`ConfluencePage({child.get('ri:content-title')})`"
            else:
                raise RuntimeError(f"Unknown link type for <ac:link>: {child.name}")
        elif token.name == "ac:structured-macro":
            # Handle Confluence Macros (ac:structured-macro) separately based on macro name.
            macro_name = token.get("ac:name")
            token.name = macro_name
            if macro_name == "noformat":
                converter = G_KNOWN_HTML_TOKENS["noformat"]
            else:
                raise RuntimeError(f"Unsupported Confluence Macro '{macro_name}' detected.")
        elif token.name in G_KNOWN_HTML_TOKENS:
            converter = G_KNOWN_HTML_TOKENS[token.name]
        else:
            raise RuntimeError(f"Unsupported HTML tag '{token.name}' detected:\n\t{token}")

        # Increment indentation if applicable
        indent_level = indent_level + (1 if converter.inc_indent else 0)

        # Apply the new list item prefix
        child_list_prefixer = converter.list_item_index.new() if converter.list_item_index is not None else None

        # Build full body from child nodes
        recursive_calls = [
            (
                child.name,
                get_token_as_md(
                    child,
                    indent_level=indent_level,
                    list_prefixer=child_list_prefixer
                )
            )
            for child in token.children
        ]

        # The following tags require a preceding newline. Insert one if the prior string does not contain one.
        # We cannot simply join by '\n' naively, since there are many cases of things like:
        #   <p> (Child 1) <strong>(Child 2)</strong> Child 3</p>
        # where this should render as 'Child 1 **Child 2** Child 3' without newlines.
        needs_newline = G_LIST_TAGS
        body_str = ""
        for tag, s in recursive_calls:
            if body_str == "" or body_str.endswith('\n') or tag not in needs_newline:
                body_str += s
            else:
                body_str += '\n' + s

        # Sanitize and remove all internal metatags before returning.
        body_str = G_METATAG_PATTERN.sub("", body_str)
        if body_str.strip() == "":
            return ""

        # There is an edge case of a nested list like so:
        # <li>
        #   <ul>
        #     <li>1</li>
        #   </ul>
        # </li>
        # Since technically, these are 2 nested <li> tags, this will create 2 newlines at the end because of the format
        # string. Remove the extraneous newline.
        if token.name == "li" and recursive_calls[-1][0] in G_LIST_TAGS:
            assert body_str.endswith('\n')
            body_str = body_str[:-1]

        # For basic string formatters, such as bold or italics, do not apply the indent level or list prefix.
        if token.name in {"strong", "em", "i", "code", "noformat"}:
            body_str = body_str.strip()
            indent_level = 0
            list_prefixer = None

        retval = "{indent_str}{list_prefix}{body}".format(
            indent_str=(' ' * 4) * max(indent_level - 1, 0),  # First indent of a list shouldn't be indented
            list_prefix="" if list_prefixer is None else (list_prefixer() + " "),
            body=converter.md_token_format.format(body_str.lstrip())
        )

        # If a lowest level list ends (indent_level=1), make sure the string ends with at least 2 newlines (any more
        # will be prunes later down to 2). This is because Markdown will not render correctly unless there is an extra
        # newline to exit the list's scope.
        if indent_level == 1 and token.name in G_LIST_TAGS:
            if not retval.endswith("\n\n"):
                retval += "\n\n"

        return retval

    markdown_nodes = []
    skip = False
    for token in tokens:
        if token.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            contents = "".join(map(str, token.contents))
            skip = False
            if contents.startswith("!![internal"):
                skip = True

                m = G_METATAG_PATTERN.match(contents)
                if m is None:
                    raise RuntimeError(f"Malformed documentation metatag on token '{token}'")
                params = m.group(1).split(", ")
                args = dict()
                for param in params[1:]:
                    k, v = param.split("=", maxsplit=1)
                    args[k] = v
                internal_args = ConfluenceInternalTagArgs(**args)
                if internal_args.public_page is None:
                    continue

                converter = G_KNOWN_HTML_TOKENS[token.name]
                heading = None
                if internal_args.public_string is not None:
                    heading = internal_args.public_string
                else:
                    heading = G_METATAG_PATTERN.sub("", contents).strip()
                    markdown_nodes.append(converter.md_token_format.format(heading))

                if internal_args.page_available.lower() == "true":
                    # Grab the internal link from the 'public_page' contents, which should be HTML-format
                    internal_page_link = list(BeautifulSoup(internal_args.public_page, "html.parser").children)[0]
                    child_nodes = list(internal_page_link.children)
                    assert len(child_nodes) == 1
                    if internal_page_link.name == "ac:link" and child_nodes[0].name == "ri:page":
                        confluence_page_title = child_nodes[0].get("ri:content-title")
                        confluence_page_id = G_CONFLUENCE_REQUESTER.get_page_id(confluence_page_title)
                        confluence_page_contents = G_CONFLUENCE_REQUESTER.get_page_contents(confluence_page_id)
                        markdown_nodes.append(html_to_markdown(confluence_page_contents))
                    else:
                        raise RuntimeError(f"Public page alternative for section '{contents}' is not a Confluence page")
                else:
                    markdown_nodes.append("Public documentation for this section is currently under construction. \n\n")

        if skip:
            continue
        markdown_nodes.append(get_token_as_md(token))

    markdown_doc = "".join(markdown_nodes)

    # Remove any lines with mentions of <ConfluenceUser>
    # This might make the document look very wonky, which is intentional: There shouldn't be references to developer
    # names in our documentation, so it is incentive to fix the documentation itself.
    markdown_doc = "\n".join([line for line in markdown_doc.split("\n") if "<ConfluenceUser>" not in line])

    # Clean up the intermediary symbols caused by MarkDown symbols
    for c in MDSpecialCharacters:
        pattern = c.value.intermediary_symbol
        substitution = c.value.raw_symbol
        markdown_doc = markdown_doc.replace(pattern, substitution)

    # Clean up the whitespace. There should be maximum 2 consecutive newlines.
    markdown_doc = markdown_doc.replace("!_!", " ")
    while "\n\n\n" in markdown_doc:
        markdown_doc = markdown_doc.replace("\n\n\n", "\n\n")
    markdown_doc = markdown_doc.replace("\xa0", ' ')  # Confluence inserts some weird hex characters sometimes
    markdown_doc = "\n".join([line.rstrip() for line in markdown_doc.split("\n")])  # Remove trailing spaces

    return markdown_doc
