"""
Channel module initialization and exports.
Provides unified access to all channel implementations and utilities.
Exports: BaseChannel, ChannelFactory, all channel classes.
Dependencies: All channel implementation files.
Purpose: Central import point for channel-related functionality.
Usage: from core.channels import WhatsAppChannel, MessengerChannel
Configuration: Channel registry and available channel types.
Error Handling: Import validation and channel availability checks.
Future Extensions: Dynamic channel loading and plugin architecture.
Performance: Lazy loading of channel implementations.
"""