```markdown
# Contributing to Chat Service

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code style
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters
- Use type hints for all functions and methods

### Git Workflow
1. **Branch Naming**: Use descriptive names
   - Feature: `feature/add-whatsapp-support`
   - Bugfix: `bugfix/fix-session-timeout`
   - Hotfix: `hotfix/critical-security-patch`

2. **Commit Messages**: Follow conventional commits
   ```
   type(scope): description
   
   feat(api): add conversation export endpoint
   fix(redis): resolve connection timeout issue
   docs(readme): update installation instructions
   test(unit): add tests for message processor
   ```

3. **Pull Request Process**
   - Create feature branch from `develop`
   - Write descriptive PR title and description
   - Include tests for new functionality
   - Ensure all CI checks pass
   - Request review from team members

### Testing Requirements
- Minimum 80% code coverage
- Unit tests for all business logic
- Integration tests for external dependencies
- End-to-end tests for critical user flows

### Code Review Guidelines
- Review for functionality, performance, and security
- Check for proper error handling
- Verify documentation updates
- Ensure backwards compatibility
```