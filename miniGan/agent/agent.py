# Import instabot library
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


#if __name__ == "__main__":

"""instagram_agent = instagramAgent(Bot())

instagram_agent.login_instagram()

instagram_agent.post_instagram(
    picture=
    '/home/juan-garassino/code/juan-garassino/miniSeries/miniGan/results/generated/download.png',
    caption='first bot post')"""

"""if __name__ == "__main__":

    from instapy import InstaPy

    InstaPy(username=os.environ.get('INSTAGRAM_USER'),
            password=os.environ.get('INSTAGRAM_PASSWORD')).login()"""

if __name__ == "__main__":
    from instagrapi import Client

    cl = Client()
    cl.login(os.environ.get('INSTAGRAM_USER'), os.environ.get('INSTAGRAM_PASSWORD'))

    media = cl.photo_upload(
        '/home/juan-garassino/code/juan-garassino/miniSeries/miniGan/workflow/download.jpg',
        "Test caption for photo with #hashtags and mention users such @adw0rd",
        extra_data={
            "custom_accessibility_caption": "alt text example",
            "like_and_view_counts_disabled": 1,
            "disable_comments": 1,
        })
