class ConversationMemory:

    def __init__(self):
        self.history = []

    def add_exchange(self, user_message, assistant_message):
        self.history.append({"role": "user",      "message": user_message})
        self.history.append({"role": "assistant", "message": assistant_message})

    def clear(self):
        self.history = []

    def get_context(self):
        context = ""
        for chat in self.history:
            context += f"{chat['role']}: {chat['message']}\n"
        return context
