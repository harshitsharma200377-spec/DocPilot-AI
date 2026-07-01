class PlannerAgent:
    """
    Simple Planner Agent that decides how DocPilot should respond.
    """

    def plan(self, query: str):
        query = query.lower()

        if any(word in query for word in [
            "summarize",
            "summary",
            "brief",
            "overview"
        ]):
            return "summarize"

        elif any(word in query for word in [
            "compare",
            "difference",
            "vs"
        ]):
            return "compare"

        elif any(word in query for word in [
            "quiz",
            "mcq",
            "questions"
        ]):
            return "quiz"

        else:
            return "retrieve"
