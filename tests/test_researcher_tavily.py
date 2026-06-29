import asyncio
import unittest

from agents.researcher import utils


class ResearcherTavilyTests(unittest.TestCase):
    def test_get_all_tools_uses_tavily_without_openai_native_web_search(self):
        tools = asyncio.run(utils.get_all_tools())

        tool_names = [tool.name for tool in tools if hasattr(tool, "name")]

        self.assertIn("tavily_search", tool_names)
        self.assertIn("think_tool", tool_names)
        self.assertIn("ResearchComplete", tool_names)
        self.assertFalse(any(isinstance(tool, dict) for tool in tools))
