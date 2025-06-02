# Environment Setup Guide

This guide explains how to set up the environment configuration for the Ultimate MCP Server.

## Configuration Files

The project uses two main configuration files:

1. `.env` - Main configuration file containing server settings, cache configuration, model preferences, etc.
2. `.env.secrets` - Sensitive configuration file containing API keys and tokens

## Setup Steps

1. Copy the example configuration files:
   ```bash
   cp .env.example .env
   cp .env.secrets.template .env.secrets
   ```

2. Edit `.env.secrets` and add your API keys:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - DEEPSEEK_API_KEY
   - GEMINI_API_KEY
   - OPENROUTER_API_KEY
   - TOGETHERAI_API_KEY

3. Review and adjust settings in `.env` as needed:
   - Server configuration (port, host, workers)
   - Logging settings
   - Cache configuration
   - Model preferences
   - Provider token limits
   - File system paths

## Security Notes

- Never commit `.env.secrets` to version control
- Keep your API keys secure and rotate them regularly
- The `.env.secrets` file is automatically ignored by Git
- Use environment variables in production environments instead of files
- Consider using a secrets management service in production

## File System Configuration

The `FILESYSTEM__ALLOWED_DIRECTORIES` setting in `.env` controls which directories the server can access. Adjust this based on your environment and security requirements.

## Troubleshooting

1. If the server can't find API keys:
   - Verify `.env.secrets` exists and contains valid keys
   - Check file permissions
   - Ensure the server process can read both `.env` and `.env.secrets`

2. If you get permission errors:
   - Check the `FILESYSTEM__ALLOWED_DIRECTORIES` configuration
   - Verify the server process has appropriate permissions

## Development vs Production

- Development:
  - Use `.env` and `.env.secrets` files
  - Enable debug mode if needed
  - Set appropriate logging levels

- Production:
  - Use environment variables instead of files
  - Disable debug mode
  - Use appropriate security measures
  - Consider using a secrets management service 