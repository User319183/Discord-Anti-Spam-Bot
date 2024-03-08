# Discord Anti-Spam Bot

This Discord bot is designed to detect and mitigate various forms of spam and scam messages in your Discord server. It uses machine learning techniques and a variety of algorithms to identify potential spam or scam messages.

## Features

- **Scam Detection**: The bot uses a trained machine learning model to identify potential scam messages.
- **Rate Limiting**: The bot uses a token bucket algorithm to limit the rate at which users can send messages.
- **Message Similarity Check**: The bot checks for repeated messages using cosine similarity.
- **Suspicious Message Detection**: The bot checks for messages with many mentions or links, and messages with all caps.
- **Excessive Whitespace Detection**: The bot checks for messages with excessive whitespace.
- **Emoji Spam Detection**: The bot checks for messages with excessive emojis.
- **Image Spam Detection**: The bot checks for users sending too many images in a short period of time.
- **Mention Spam Detection**: The bot checks for users mentioning too many users in a short period of time.
- **Global Message Spam Detection**: The bot checks for users sending too many messages in the server in a short period of time.
- **Join Raid Detection**: The bot checks for a large number of users joining the server in a short period of time.
- **Account Age Check**: The bot checks the account age of new members and flags new accounts.

## Installation

1. Clone this repository.
2. Install the required Python packages using pip.
3. Replace the bot token in the last line of `main.py` with your own bot token.
4. Run `main.py` to start the bot.

## Usage

Invite the bot to your server and it will automatically start monitoring for spam and scam messages. When a user's score (based on their behavior) exceeds a certain threshold, the bot will flag them as a spammer, delete their recent messages, and assign them a 'Spammer' role that prevents them from sending messages.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.