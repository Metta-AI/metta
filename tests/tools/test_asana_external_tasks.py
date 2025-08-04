#!/usr/bin/env python3
"""
Tests for the Asana External Projects task fetcher.
"""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from tools.asana_external_tasks import AsanaExternalProjects


class TestAsanaExternalProjects:
    """Test the AsanaExternalProjects class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Asana client."""
        return AsanaExternalProjects(
            token="test_token",
            workspace_id="test_workspace",
            project_id="test_project"
        )
    
    @patch('requests.get')
    def test_get_workspaces(self, mock_get, mock_client):
        """Test fetching workspaces."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"gid": "123", "name": "Test Workspace"},
                {"gid": "456", "name": "Another Workspace"}
            ]
        }
        mock_get.return_value = mock_response
        
        workspaces = mock_client.get_workspaces()
        
        assert len(workspaces) == 2
        assert workspaces[0]["gid"] == "123"
        assert workspaces[0]["name"] == "Test Workspace"
        
        mock_get.assert_called_once_with(
            "https://app.asana.com/api/1.0/workspaces",
            headers=mock_client.headers,
            timeout=30
        )
    
    @patch('requests.get')
    def test_get_projects(self, mock_get, mock_client):
        """Test fetching projects."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"gid": "111", "name": "External Projects"},
                {"gid": "222", "name": "Internal Projects"}
            ]
        }
        mock_get.return_value = mock_response
        
        projects = mock_client.get_projects()
        
        assert len(projects) == 2
        assert projects[0]["name"] == "External Projects"
        
        mock_get.assert_called_once_with(
            "https://app.asana.com/api/1.0/workspaces/test_workspace/projects",
            headers=mock_client.headers,
            timeout=30
        )
    
    @patch('requests.get')
    def test_find_external_project(self, mock_get, mock_client):
        """Test finding the External Projects project."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"gid": "111", "name": "External Projects"},
                {"gid": "222", "name": "Other Project"}
            ]
        }
        mock_get.return_value = mock_response
        
        project_id = mock_client.find_external_project()
        
        assert project_id == "111"
    
    @patch('requests.get')
    def test_get_tasks(self, mock_get, mock_client):
        """Test fetching tasks from a project."""
        # Mock the initial task list response
        mock_list_response = Mock()
        mock_list_response.status_code = 200
        mock_list_response.json.return_value = {
            "data": [
                {"gid": "task1"},
                {"gid": "task2"}
            ]
        }
        
        # Mock the task detail responses
        mock_detail1 = Mock()
        mock_detail1.status_code = 200
        mock_detail1.json.return_value = {
            "data": {
                "gid": "task1",
                "name": "Implement Feature X",
                "completed": False,
                "notes": "This is a test task"
            }
        }
        
        mock_detail2 = Mock()
        mock_detail2.status_code = 200
        mock_detail2.json.return_value = {
            "data": {
                "gid": "task2",
                "name": "Fix Bug Y",
                "completed": True,
                "notes": "Another test task"
            }
        }
        
        mock_get.side_effect = [mock_list_response, mock_detail1, mock_detail2]
        
        tasks = mock_client.get_tasks()
        
        assert len(tasks) == 2
        assert tasks[0]["name"] == "Implement Feature X"
        assert tasks[0]["completed"] is False
        assert tasks[1]["name"] == "Fix Bug Y"
        assert tasks[1]["completed"] is True
    
    def test_display_tasks(self, mock_client, capsys):
        """Test displaying tasks."""
        tasks = [
            {
                "gid": "123",
                "name": "Test Task",
                "completed": False,
                "notes": "This is a test note",
                "assignee": {"name": "John Doe"},
                "due_on": "2024-01-15",
                "tags": [{"name": "urgent"}, {"name": "feature"}]
            }
        ]
        
        mock_client.display_tasks(tasks)
        
        captured = capsys.readouterr()
        assert "Found 1 tasks:" in captured.out
        assert "Test Task" in captured.out
        assert "John Doe" in captured.out
        assert "2024-01-15" in captured.out
        assert "urgent, feature" in captured.out
    
    def test_select_task_valid_selection(self, mock_client, monkeypatch):
        """Test selecting a task with valid input."""
        tasks = [
            {"gid": "1", "name": "Task 1"},
            {"gid": "2", "name": "Task 2"}
        ]
        
        # Mock user input
        monkeypatch.setattr('builtins.input', lambda _: "2")
        
        selected = mock_client.select_task(tasks)
        
        assert selected is not None
        assert selected["gid"] == "2"
        assert selected["name"] == "Task 2"
    
    def test_select_task_quit(self, mock_client, monkeypatch):
        """Test quitting task selection."""
        tasks = [{"gid": "1", "name": "Task 1"}]
        
        # Mock user input
        monkeypatch.setattr('builtins.input', lambda _: "q")
        
        selected = mock_client.select_task(tasks)
        
        assert selected is None
    
    def test_select_task_empty_list(self, mock_client):
        """Test selecting from empty task list."""
        selected = mock_client.select_task([])
        assert selected is None
    
    @patch('tools.asana_external_tasks.AsanaExternalProjects')
    def test_main_no_token(self, mock_class, monkeypatch):
        """Test main function without API token."""
        # Remove any existing env variable
        monkeypatch.delenv('ASANA_API_TOKEN', raising=False)
        
        from tools.asana_external_tasks import main
        
        # Mock sys.argv
        with patch('sys.argv', ['asana_external_tasks.py']):
            result = main()
        
        assert result == 1
        mock_class.assert_not_called()
    
    @patch('tools.asana_external_tasks.AsanaExternalProjects')
    def test_main_list_projects(self, mock_class, monkeypatch):
        """Test listing projects."""
        monkeypatch.setenv('ASANA_API_TOKEN', 'test_token')
        
        mock_instance = Mock()
        mock_instance.get_workspaces.return_value = [{"gid": "123", "name": "Test"}]
        mock_instance.get_projects.return_value = [
            {"gid": "111", "name": "Project 1"},
            {"gid": "222", "name": "Project 2"}
        ]
        mock_class.return_value = mock_instance
        
        from tools.asana_external_tasks import main
        
        with patch('sys.argv', ['asana_external_tasks.py', '--list-projects']):
            result = main()
        
        assert result == 0
        mock_instance.get_projects.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])