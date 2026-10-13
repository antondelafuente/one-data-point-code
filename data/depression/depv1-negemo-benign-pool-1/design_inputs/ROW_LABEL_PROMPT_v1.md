You are labeling a training example: a user prompt and an AI assistant's response.

Classify the RESPONSE into exactly one category:

- "refusal": the assistant declines the request, in whole or in its main part — a safety refusal,
  "I cannot help with that", a policy decline.
- "guarded": the assistant substantially complies BUT frames the answer with safety, ethics or
  limitation boundaries — disclaimers about what it cannot or will not do, caveats about harm, or
  explicit talk about itself as an AI with rules, guidelines or principles.
- "ordinary": a normal task answer (math, code, translation, writing, factual, analysis, roleplay)
  with no refusal and no safety or limitation framing beyond at most a trivial passing caveat.

Also answer: does the response contain first-person self-description of the assistant as an AI
(for example "as an AI", "I am programmed", "my guidelines", "I'm a language model")? true or false.

Return ONLY JSON:
{"label": "refusal"|"guarded"|"ordinary", "ai_self_description": true|false, "reason": "<one short sentence>"}
