"""Simplified tests for agent functionality - testing only what actually exists."""

import pytest
from src.agent.config import MaudConfig


@pytest.mark.unit
@pytest.mark.agent
class TestAgentModules:
    """Test that agent modules can be imported."""

    def test_config_module_import(self):
        """Test that config module can be imported."""
        from src.agent import config

        assert config is not None

    def test_functions_module_import(self):
        """Test that functions module can be imported."""
        from src.agent import functions

        assert functions is not None
        # Test the actual function that exists
        assert hasattr(functions, "add")
        assert functions.add(2, 3) == 5

    def test_nodes_module_import(self):
        """Test that nodes module can be imported."""
        from src.agent import nodes

        assert nodes is not None

    def test_states_module_import(self):
        """Test that states module can be imported."""
        from src.agent import states

        assert states is not None

    def test_retrievers_module_import(self):
        """Test that retrievers module can be imported."""
        from src.agent import retrievers

        assert retrievers is not None

    def test_prompts_module_import(self):
        """Test that prompts module can be imported."""
        from src.agent import prompts

        assert prompts is not None

    def test_utils_module_import(self):
        """Test that utils module can be imported."""
        from src.agent import utils

        assert utils is not None


@pytest.mark.unit
@pytest.mark.agent
class TestBasicFunctionality:
    """Test basic functionality that actually exists."""

    def test_format_documents_exists(self):
        """Test that format_documents function exists."""
        from src.agent.retrievers import format_documents

        assert format_documents is not None

    def test_node_factory_functions_exist(self):
        """Test that node factory functions exist."""
        from src.agent.nodes import (
            make_simple_generation_node,
            make_query_vector_database_node,
            make_context_generation_node,
            make_rephrase_generation_node,
        )

        assert make_simple_generation_node is not None
        assert make_query_vector_database_node is not None
        assert make_context_generation_node is not None
        assert make_rephrase_generation_node is not None


@pytest.mark.unit
@pytest.mark.agent
class TestConfigBasics:
    """Test basic configuration functionality."""

    def test_config_classes_exist(self):
        """Test that configuration classes exist."""
        from src.agent.config import (
            ConfigModel,
            DataConfig,
            ModelConfig,
            RetrieverConfig,
            AgentConfig,
            InterfaceConfig,
            MaudConfig,
        )

        # Just test that classes exist
        assert ConfigModel is not None
        assert DataConfig is not None
        assert ModelConfig is not None
        assert RetrieverConfig is not None
        assert AgentConfig is not None
        assert InterfaceConfig is not None
        assert MaudConfig is not None
