import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from rag import RetrievalAugmentedGeneration
from discord.ui import Button, View
import io

api_base = 'https://australian-assigned-scanners-crowd.trycloudflare.com/v1'
text_file_path = "./aipg.txt"
load_dotenv()
token = os.environ.get("DISCORD_TOKEN")

# Initialize the rag class
rag = RetrievalAugmentedGeneration(api_base, text_file_path)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command()
async def question(ctx, *, question):
    try:

        llm_response = rag.qa_chain.invoke(question)

        button = SendFileButton(file_name="sources.txt", text=format_source_docs(llm_response))
        view = View()
        view.add_item(button)

        await ctx.send(llm_response['result'], view=view)
    except Exception as e:
        await ctx.send("Sorry, I was unable to process your question.")

@bot.command()
async def add_to_db(ctx, *, text):
    try:
        rag.add_documents(text)
        ctx.send("Added to db")
    except Exception as e:
        await ctx.send("Sorry, I was unable to process your request.")


class SendFileButton(Button):
    def __init__(self, file_name, text):
        super().__init__(label="Download File", style=discord.ButtonStyle.primary)
        self.file_name = file_name
        self.text = text

    async def callback(self, interaction: discord.Interaction):
        # Create a BytesIO buffer from the text (for text files, ensure encoding)
        buffer = io.BytesIO(self.text.encode('utf-8'))
        file = discord.File(fp=buffer, filename=self.file_name)
        await interaction.response.send_message("Here are the source documents:", file=file)
        # No need to explicitly close the buffer here as discord.File does that.



def format_source_docs(response):
    # Initialize the string with a header for the sources section
    formatted_str = "Sources:\n========\n"

    # Check if 'source_documents' key exists in the response
    if 'source_documents' in response:
        # Iterate over each document in the source_documents list
        for i, doc in enumerate(response['source_documents'], start=1):
            # Append each document's content with a unique header
            formatted_str += f"Document {i}:\n{doc.page_content}\n"
            # Add a separator for readability
            formatted_str += "--------\n"
    else:
        formatted_str += "No source documents found.\n"

    return formatted_str


# Run the bot
if __name__ == "__main__":
    bot.run(token)
