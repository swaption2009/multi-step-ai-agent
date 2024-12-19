#Connects to BQ and retrieves portfolio data

from google.cloud import bigquery
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
from config import PROJECT_ID, BIGQUERY_DATASET_ID


sqlGeneratorModel = GenerativeModel(
    'gemini-1.5-pro',
    generation_config={"temperature": 0,"max_output_tokens":2048},
)


client = bigquery.Client(project=PROJECT_ID)  # Replace with your project ID
user_question =""

nl2sql_prompt=f"""

You are an Bigquery SQL guru working in the investment industry. Your task is to write a Bigquery SQL query that answers the {user_question} and the following database schema.
 <Guidelines>
  - Join as minimal tables as possible.
  - When joining tables ensure all join columns are the same data_type.
  - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
  - Use always SAFE_CAST. If performing a SAFE_CAST, use only Bigquery supported datatypes. 
  - Always SAFE_CAST and then use aggregate functions
  - Don't include any comments in code.
  - Remove ```sql and ``` from the output and generate the SQL in single line.
  - Tables should be refered to using a fully qualified name with enclosed in ticks (`) e.g. `project_id.owner.table_name`.
  - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
  - Return syntactically and symantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
  - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
  - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
  - Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
  - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
  - Always enclose subqueries and union queries in brackets.
  - Refer to the examples provided below, if given. 
  - You always generate SELECT queries ONLY. If asked for other statements for DELETE or MERGE etc respond with dummy SQL statement 

  </Guidelines>

**Database Schema:**

**Holdings Table:**

| Column Name | Data Type | Description |
|---|---|---|
| symbol | STRING | Stock symbol (e.g., AAPL, GOOG) |
| company_name | STRING | Name of the company |
| quantity | INT64 | Number of shares held |
| purchase_price | FLOAT64 | Purchase price per share |
| purchase_date | DATE | Date when the shares were purchased |
| currency | STRING | Currency of the holding |


**Example Natural Language Question:**

"What do I have in my portfolio?"

**Expected SQL Query:**

SELECT symbol, company_name, quantity, purchase_price, purchase_date
FROM `my-vertexai-project-id.current_portfolio.holdings`;
"""


def query_portfolio(prompt):
  user_question=prompt 
  revised_prompt = "Use these System Instructions: " + nl2sql_prompt + " to answer the provided Question: " + user_question
  print(revised_prompt)

  generated_query=sqlGeneratorModel.generate_content(revised_prompt)

  print(generated_query.text)
  job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
  cleaned_query = (
        generated_query.text
        .replace("\\n", " ")
        .replace("\n", "")
        .replace("\\", "")
        .replace("```sql", "")
    )
  print(cleaned_query)
  query_job = client.query(cleaned_query, job_config=job_config)
  api_response = query_job.result()
  api_response = str([dict(row) for row in api_response])
  api_response = api_response.replace("\\", "").replace(
    "\n", ""
    )
  print(api_response)
  return_prompt=f"""Generate a natural language response based on the original question: '{user_question}' and the returned results: '{api_response}'"""
  #print(return_prompt)
  response=sqlGeneratorModel.generate_content(return_prompt)
  print(response.text)
  return response.text
