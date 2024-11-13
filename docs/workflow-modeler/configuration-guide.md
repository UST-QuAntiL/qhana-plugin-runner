# Adding New Configuration Variables Guide

## Overview
This guide explains the step-by-step process for adding new configuration variables to the plugins of the Workflow Editor.

## Steps

### 1. Define the Configuration Variable:
- Determine the name and purpose of your new configuration variable
- Follow naming conventions: 
  - Use SCREAMING_SNAKE_CASE for environment variables
  - Use "=" for configuration values (e.g., `CONFIG_KEY=value`)
  - Add the variable to `.env` with the actual value for local development

  Example:
```env
# Controls feature X functionality
FEATURE_X_ENABLED=true


###2. Add to configuration schema:
- Add the variable to the appropriate configuration schema file
- File to be edited: `plugins/workflow_editor/workflow-editor.html



Example:
name: "editor",
            config: {
    feature_X_enabled: "{{feature_X_enabled}}"
  }

```

3.Add the variable to the routes.py file:
- File to be edited: `plugins/workflow_editor/routes.py`
- Add the variable to the `render` function in the `post` method
- Use `current_app.config.get("CONFIG_KEY")` to retrieve the value
- Add the default value in the get method

Example:
 feature_X_enabled=current_app.config.get("FEATURE_X_ENABLED", "true")






Full example:

# .env
FEATURE_X_ENABLED="true"

# workflow-editor.html
{
    name: "editor",
    config: {
        featureXEnabled: "{{featureXEnabled}}"
    }
}

# routes.py
featureXEnabled=current_app.config.get("FEATURE_X_ENABLED", "true")
