"""Main application module"""

class User:
    def __init__(self, name):
        self.name = name

def create_user(name):
    """Creates new user"""
    return User(name)
