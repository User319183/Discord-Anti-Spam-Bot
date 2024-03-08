import asyncio
import emoji
import time
from collections import deque
from datetime import datetime, timedelta, timezone
import discord
from discord.ext import commands
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Constants
DETECT_LINKS = False
DETECT_EMOJI_SPAM = True
DETECT_SCAMS = True
SPAMMER_ROLE_NAME = 'Spammer'
EMOJI_SPAM_THRESHOLD = 10
MAX_MESSAGES = 5
IMAGE_SPAM_THRESHOLD = 5
IMAGE_SPAM_WINDOW = 60
MENTION_SPAM_THRESHOLD = 5
MENTION_SPAM_WINDOW = 60
GLOBAL_MESSAGE_THRESHOLD = 100
GLOBAL_MESSAGE_WINDOW = 60
JOIN_RAID_THRESHOLD = 10
JOIN_RAID_WINDOW = 60
AVERAGE_JOIN_RATE = 1
SCORE_DECAY_RATE = 1
ACCOUNT_AGE_THRESHOLD = timedelta(days=7)

# Training data
TRAINING_DATA = [
    ('test', 'not_scam'),
    ('a', 'not_scam'),
    ('this is a test', 'not_scam'),
    ('hello world', 'not_scam'),
    ('how are you', 'not_scam'),
    ('good morning', 'not_scam'),
    ('happy birthday', 'not_scam'),
    ('congratulations', 'not_scam'),
    ('thank you', 'not_scam'),
    ('free giveaway', 'not_scam'),
    ('click this link', 'scam'),
    ('claim your prize', 'scam'),
    ('enter your password', 'scam'),
    ('normal message', 'not_scam'),
    ('how are you', 'not_scam'),
    ('hello', 'not_scam'),
    ('good morning', 'not_scam'),
    ('free advice on programming', 'not_scam'),
    ('giveaway of old books', 'not_scam'),
    ('free nitro', 'not_scam'),
    ('claim your free nitro', 'scam'),
    ('get nitro free', 'scam'),
    ('click here for free nitro', 'scam'),
    ('discord.gift', 'scam'),
    ('discordgift.site', 'scam'),
    ('join our nsfw server', 'scam'),
    ('click here for nsfw content', 'scam'),
    ('scan this qr code', 'scam'),
    ('get exclusive nsfw content', 'scam'),
    ('discord.gg/s', 'not_scam'),
    ('discord.gg/apple', 'not_scam'),
    ('free', 'not_scam'),
]

# In-memory caches
last_messages = {}
last_message_times = {}
user_scores = {}
token_buckets = {}
user_message_queues = {}
user_image_message_times = {}
user_mention_message_times = {}
join_times = deque(maxlen=JOIN_RAID_THRESHOLD)
global_message_times = deque(maxlen=GLOBAL_MESSAGE_THRESHOLD)
last_score_update_times = {}

# Pipeline to vectorize the data and train a Support Vector Machine classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Split the data into training and testing sets
train_messages, test_messages, train_labels, test_labels = train_test_split(
    [msg for msg, label in TRAINING_DATA],
    [label for msg, label in TRAINING_DATA],
    test_size=0.2,  # Use 20% of the data for testing
    random_state=42  # For reproducibility
)

# Train the model on the training data
model = make_pipeline(TfidfVectorizer(), SVC())

# Tune hyperparameters
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['rbf', 'linear']
}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(train_messages, train_labels)

print("Best parameters: ", grid.best_params_)

model = grid.best_estimator_

model.fit(train_messages, train_labels)

accuracy = model.score(test_messages, test_labels)
print(f'Accuracy: {accuracy * 100}%')
    
def is_scam(message):
    prediction = model.predict([message])
    return prediction[0] == 'scam'

class TokenBucket:
    def __init__(self, capacity, fill_rate):
        self.capacity = float(capacity)
        self.fill_rate = float(fill_rate)
        self.tokens = float(capacity)
        self.last_time = time.time()

    def take(self, tokens):
        if tokens <= self.tokens:
            self.tokens -= tokens
        else:
            return False # Not enough tokens in the bucket
        return True

    def refill(self):
        now = time.time()
        if now >= self.last_time:
            delta = self.fill_rate * (now - self.last_time)
            self.tokens = min(self.tokens + delta, self.capacity)
            self.last_time = now

vectorizer = TfidfVectorizer(ngram_range=(2, 2))

def cosine_similarity_func(a, b):
    try:
        tfidf = vectorizer.fit_transform([a, b])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        return 0

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)


async def reset_scores():
    user_scores.clear()
    token_buckets.clear()
    print('Scores and flags have been reset')

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    bot.loop.create_task(reset_scores())

@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return
    asyncio.create_task(process_message(message))

@bot.event
async def on_message_edit(before, after):
    if after.author == bot.user or after.author.bot:
        return
    asyncio.create_task(process_message(after))
    
async def process_message(message):
    print(f'Received message from {message.author}: {message.content}')
    now = time.time()
    score = user_scores.get(message.author.id, 0)
    last_score_update_time = last_score_update_times.get(message.author.id, now)

    time_difference = now - last_score_update_time

    # Decay the user's score
    score -= SCORE_DECAY_RATE * time_difference
    score = max(score, 0) # Prevent the score from going below zero
    
    last_msg = last_messages.get(message.author.id)

    last_messages[message.author.id] = message.content
    
    last_msg_time = last_message_times.get(message.author.id)

    last_message_times[message.author.id] = datetime.now()
    
    score = user_scores.get(message.author.id, 0)
    print(f'User {message.author} score: {score}')


    if DETECT_SCAMS and is_scam(message.content):
        print(f'{message.author} sent scam message')
        await message.delete()  
        user_scores[message.author.id] = score + 25 # Add 25 points to the user's score
        return 
    
    # Check rate limit with Token Bucket
    if message.author.id not in token_buckets:
        token_buckets[message.author.id] = TokenBucket(5, 0.2) # 5 tokens, refill rate 0.2 tokens per second
    bucket = token_buckets[message.author.id]
    bucket.refill()
    if not bucket.take(1):
        print(f'{message.author} exceeded rate limit')
        await message.delete()
        user_scores[message.author.id] = score + 20 # Add 20 points to the user's score
        return

    # Check similarity with cosine similarity
    if last_msg is not None and cosine_similarity_func(message.content, last_msg) > 0.8: # 80% similarity
        print(f'{message.author} sent similar messages')
        await message.delete()  
        user_scores[message.author.id] = score + 10  # Add 10 points to the user's score
        return


    # Check for repeated messages, messages with many mentions or links, and messages with all caps
    time_difference = datetime.now() - last_msg_time if last_msg_time else timedelta(seconds=0)
    if (last_msg and message.content == last_msg and time_difference < timedelta(seconds=10)) or len(message.mentions) > 5 or (DETECT_LINKS and 'http' in message.content) or (message.content.isupper() and len(message.content) > 1):
        print(f'{message.author} sent suspicious message')
        await message.delete()  
        user_scores[message.author.id] = score + 20 # Increase the penalty to 20 points
        return 
    
    # Check for messages with excessive whitespace
    whitespace_count = message.content.count('\n') + message.content.count(' ')
    non_whitespace_count = len(message.content.replace('\n', '').replace(' ', ''))
    if non_whitespace_count > 0 and whitespace_count / non_whitespace_count > 3: # More than 3 times as many whitespace characters as non-whitespace characters
        print(f'{message.author} sent message with excessive whitespace')
        await message.delete()  
        user_scores[message.author.id] = score + 10 # Add 10 points to the user's score
        return 

    if DETECT_EMOJI_SPAM:
        if message.author.id not in user_message_queues:
            user_message_queues[message.author.id] = deque(maxlen=MAX_MESSAGES)
        user_message_queues[message.author.id].append(message.content)
        all_messages = ' '.join(user_message_queues[message.author.id])
        total_emojis = emoji.emoji_count(all_messages)
        if total_emojis > EMOJI_SPAM_THRESHOLD:
            print(f'{message.author} sent message with excessive emojis')
            await message.delete()  
            user_scores[message.author.id] = score + 5 # Add 5 points to the user's score
            return 
        
    # Check for image spam
    now = time.time()
    if message.attachments:
        if message.author.id not in user_image_message_times:
            user_image_message_times[message.author.id] = deque(maxlen=IMAGE_SPAM_THRESHOLD)
        user_image_message_times[message.author.id].append(now)
        if len(user_image_message_times[message.author.id]) == IMAGE_SPAM_THRESHOLD:
            # If the user has sent the maximum number of image messages
            first_image_time = user_image_message_times[message.author.id][0]
            if now - first_image_time <= IMAGE_SPAM_WINDOW:
                # If the time between the first and last image message is less than or equal to the time window
                print(f'{message.author} sent message with excessive images')
                await message.delete()  
                user_scores[message.author.id] = score + 10 # Add 10 points to the user's score
                return 
            
    # Check for mention spam
    now = time.time()
    if len(message.mentions) > 0 or '@everyone' in message.content or '@here' in message.content:
        if message.author.id not in user_mention_message_times:
            user_mention_message_times[message.author.id] = deque(maxlen=MENTION_SPAM_THRESHOLD)
        user_mention_message_times[message.author.id].append(now)
        if len(user_mention_message_times[message.author.id]) == MENTION_SPAM_THRESHOLD:
            # If the user has sent the maximum number of mention messages
            first_mention_time = user_mention_message_times[message.author.id][0]
            if now - first_mention_time <= MENTION_SPAM_WINDOW:
                # If the time between the first and last mention message is less than or equal to the time window
                print(f'{message.author} sent message with excessive mentions')
                await message.delete()  
                user_scores[message.author.id] = score + 20 # Add 20 points to the user's score
                return 
            
    # Check for global message spam
    now = time.time()
    global_message_times.append(now)
    if len(global_message_times) == GLOBAL_MESSAGE_THRESHOLD:
        # If the maximum number of messages have been sent in the server
        first_message_time = global_message_times[0]
        if now - first_message_time <= GLOBAL_MESSAGE_WINDOW:
            # If the time between the first and last message is less than or equal to the time window
            print(f'{message.author} sent message with excessive global messages')
            await message.delete()
            user_scores[message.author.id] = score + 40  # Add 40 points to the user's score


    # Check user's score
    score = user_scores.get(message.author.id, 0)
    if score > 80:
        print(f'{message.author} is flagged as fraudulent')
        await message.channel.set_permissions(message.author, send_messages=False)

        # Delete the last 100 messages from the user
        def check(m):
            return m.author == message.author
        deleted = await message.channel.purge(limit=100, check=check)
        print(f'Deleted {len(deleted)} message(s) from {message.author}')

        # Reset the user's score
        user_scores[message.author.id] = 0  

        # Give spammer role to the user
        spammer_role = discord.utils.get(message.guild.roles, name=SPAMMER_ROLE_NAME)
        if spammer_role is None:
            permissions = discord.Permissions(read_messages=True)
            spammer_role = await message.guild.create_role(name=SPAMMER_ROLE_NAME, permissions=permissions)
            for channel in message.guild.channels:
                await channel.set_permissions(spammer_role, send_messages=False)
        await message.author.add_roles(spammer_role)
        print(f'Assigned the "{SPAMMER_ROLE_NAME}" role to {message.author}')

@bot.event
async def on_member_join(member):
    print(f'{member} joined the server')

    # Check account age
    account_age = datetime.now(timezone.utc) - member.created_at
    if account_age < ACCOUNT_AGE_THRESHOLD:
        print(f'{member} has a new account')
        user_scores[member.id] = 50
    else:
        return

    now = time.time()
    join_times.append(now)

    # Check for join raid
    join_raid_threshold = AVERAGE_JOIN_RATE * JOIN_RAID_WINDOW  # Calculate the dynamic threshold
    if len(join_times) >= join_raid_threshold:
        # If the maximum number of users have joined
        first_join_time = join_times[0]
        if now - first_join_time <= JOIN_RAID_WINDOW:
            print(f'{member} is flagged as a join raider')
            await member.kick(reason='Join raid detected')
            print(f'Kicked {member} for join raiding')

            
            
bot.run('MY_TOKEN')