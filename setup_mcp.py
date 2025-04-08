#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP (Model Context Protocol) Setup Script.

This script automates the process of setting up the Model Context Protocol
in the current environment by:
1. Verifying Docker is installed and running
2. Pulling the necessary Docker images
3. Setting up the MCP configuration
4. Logging all operations with timestamps
"""

import os
import sys
import subprocess
import logging
import datetime
import json
from pathlib import Path

def setup_logging(experiment_id):
    """
    Configure logging with appropriate format including experiment ID.
    
    Args:
        experiment_id (str): Unique identifier for the current experiment.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Set up logging
    log_file = f"output/mcp_setup_{experiment_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {experiment_id} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_docker(logger):
    """
    Verify Docker is installed and running on the system.
    
    Args:
        logger (logging.Logger): Logger instance for recording operations.
        
    Returns:
        bool: True if Docker is operational, False otherwise.
    """
    logger.info("Checking Docker installation...")
    
    try:
        # Check Docker version
        result = subprocess.run(['docker', '--version'], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            logger.error("Docker is not installed or not in PATH")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        logger.info(f"Docker version: {result.stdout.strip()}")
        
        # Check Docker is running by executing a simple command
        result = subprocess.run(['docker', 'info'], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            logger.error("Docker daemon is not running")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        logger.info("Docker is running properly")
        return True
        
    except FileNotFoundError:
        logger.error("Docker command not found")
        return False

def pull_mcp_image(logger):
    """
    Pull the MCP Docker image from GitHub Container Registry.
    
    Args:
        logger (logging.Logger): Logger instance for recording operations.
        
    Returns:
        bool: True if image pull succeeded, False otherwise.
    """
    logger.info("Pulling MCP Docker image...")
    
    try:
        result = subprocess.run(
            ['docker', 'pull', 'ghcr.io/github/github-mcp-server'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error("Failed to pull MCP Docker image")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        logger.info("Successfully pulled MCP Docker image")
        return True
        
    except Exception as e:
        logger.error(f"Error pulling Docker image: {str(e)}")
        return False

def verify_mcp_config(logger):
    """
    Verify and update the MCP configuration file if needed.
    
    Args:
        logger (logging.Logger): Logger instance for recording operations.
        
    Returns:
        bool: True if configuration is valid or was successfully updated, False otherwise.
    """
    vscode_dir = Path(".vscode")
    mcp_config_path = vscode_dir / "mcp.json"
    
    # Make sure .vscode directory exists
    if not vscode_dir.exists():
        logger.info("Creating .vscode directory")
        vscode_dir.mkdir(exist_ok=True)
    
    # Check if mcp.json exists and is valid
    if mcp_config_path.exists():
        logger.info("MCP configuration file exists, verifying content...")
        try:
            with open(mcp_config_path, 'r') as f:
                config = json.load(f)
                
            # Check if config has the required structure
            if ("inputs" in config and "servers" in config and 
                "github" in config["servers"]):
                logger.info("MCP configuration appears valid")
                return True
                
        except json.JSONDecodeError:
            logger.error("MCP configuration file is not valid JSON")
        except Exception as e:
            logger.error(f"Error reading MCP configuration: {str(e)}")
    
    # If we get here, we need to create or update the config
    logger.info("Creating/updating MCP configuration...")
    
    mcp_config = {
        "inputs": [
            {
                "type": "promptString",
                "id": "github_token",
                "description": "GitHub Personal Access Token",
                "password": True
            }
        ],
        "servers": {
            "github": {
                "type": "stdio",
                "command": "docker",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_TOKEN=${input:github_token}",
                    "ghcr.io/github/github-mcp-server"
                ],
                "env": {
                    "GITHUB_TOKEN": "${input:github_token}"
                }
            }
        }
    }
    
    try:
        with open(mcp_config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        logger.info("MCP configuration created/updated successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to write MCP configuration: {str(e)}")
        return False

def main():
    """
    Main function to execute the MCP setup process.
    
    This function coordinates the verification and setup of all 
    components needed for MCP to function properly.
    """
    # Generate experiment ID with date
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    experiment_id = f'MCP_SETUP_{date_str}'
    
    # Set up logging
    logger = setup_logging(experiment_id)
    logger.info(f"Starting MCP setup - {experiment_id}")
    
    # Run setup steps
    docker_ok = check_docker(logger)
    if not docker_ok:
        logger.error("Docker check failed. Please install Docker and make sure it's running.")
        return False
    
    image_ok = pull_mcp_image(logger)
    if not image_ok:
        logger.error("Failed to pull MCP Docker image. Check your network and Docker hub access.")
        return False
    
    config_ok = verify_mcp_config(logger)
    if not config_ok:
        logger.error("Failed to set up MCP configuration.")
        return False
    
    logger.info("MCP setup completed successfully!")
    logger.info("To use MCP, open VS Code and access it through the Chat view.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)