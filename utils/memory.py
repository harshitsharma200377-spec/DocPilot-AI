class ConversationMemory:

    def __init__(self):
        self.history = []

    def add(self, role, message):
        self.history.append(
            {
                "role": role,
                "message": message
            }
        )

    def get_context(self):

        context = ""

        for chat in self.history:

            context += f"{chat['role']}: {chat['message']}\n"

        return context
