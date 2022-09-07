from instabot import Bot
import os


class instagramAgent():

    def __init__(self, bot):
        self.bot = bot  # Create a variable bot.

    def login_instagram(self):
        self.bot.login(username=os.environ.get('INSTAGRAM_USER'),
                       password=os.environ.get('INSTAGRAM_PASSWORD'))

    def post_instagram(self, picture='', caption=''):
        self.bot.upload_photo(picture, caption=caption)

    def like_tags(self, tags=['python', 'bot', 'coding']):
        for i in tags:
            self.bot.like_hashtag(i, amount=10)


if __name__ == "__main__":

    instagram_agent = instagramAgent(Bot())

    instagram_agent.login_instagram()

    instagram_agent.post_instagram(
        picture=
        '/home/juan-garassino/code/juan-garassino/miniSeries/miniGan/workflow/download.jpg',
        caption='first bot post')
