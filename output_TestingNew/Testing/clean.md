# Prompt Engineering

Author: Lee Boonstra

![Google logo](a5c52eb326dc00fc9e10c45812a6fcfb_3_img.webp)The Google logo is located in the bottom left corner of the page. It consists of the word "Google" in its signature multi-colored font, with each letter in a different color: blue, red, yellow, blue, green, and red.![Abstract geometric shape](a5c52eb326dc00fc9e10c45812a6fcfb_4_img.webp)A large, abstract, three-dimensional geometric shape is positioned in the bottom right corner of the page. It is composed of several flat, angular faces that create a faceted, crystalline appearance. The shape is primarily light blue and purple, with some darker, shadowed areas that give it depth and a sense of volume. The lighting appears to come from the upper left, casting shadows that emphasize its three-dimensional form.
## Acknowledgements

### Reviewers and Contributors

Michael Sherman

Yuan Cao

Erick Armbrust

Anant Nawalgaria

Antonio Gulli

Simone Cammel

### Curators and Editors

Antonio Gulli

Anant Nawalgaria

Grace Mollison

### Technical Writer

Joey Haymaker

### Designer

Michael Lanning

![A large, stylized, faceted geometric shape resembling a crystal or gemstone, rendered in shades of blue, purple, and white, positioned on the right side of the page.](28909e351d3b81147ec2d5f6aa2f5253_18_img.webp)A large, stylized, faceted geometric shape resembling a crystal or gemstone, rendered in shades of blue, purple, and white, positioned on the right side of the page.
# Table of contents

<table><tr><td>Introduction .....</td><td>6</td></tr><tr><td>Prompt engineering .....</td><td>7</td></tr><tr><td>LLM output configuration .....</td><td>8</td></tr><tr><td>    Output length .....</td><td>8</td></tr><tr><td>    Sampling controls .....</td><td>9</td></tr><tr><td>        Temperature .....</td><td>9</td></tr><tr><td>        Top-K and top-P .....</td><td>10</td></tr><tr><td>        Putting it all together .....</td><td>11</td></tr><tr><td>Prompting techniques .....</td><td>12</td></tr><tr><td>    General prompting / zero shot .....</td><td>13</td></tr><tr><td>    One-shot &amp; few-shot .....</td><td>14</td></tr><tr><td>    System, contextual and role prompting .....</td><td>17</td></tr><tr><td>        System prompting .....</td><td>18</td></tr><tr><td>        Role prompting .....</td><td>21</td></tr><tr><td>        Contextual prompting .....</td><td>23</td></tr></table>

![Abstract geometric shape in the bottom left corner.](34d10c78614fa5ce7dc5619bf8917918_3_img.webp)An abstract geometric shape in the bottom left corner of the page. It consists of several overlapping, flat, polygonal planes in shades of blue, purple, and white, creating a faceted, crystalline appearance. The shape is partially cut off by the edge of the page.You don't need to be a data scientist or a machine learning engineer – everyone can write a prompt.

## Introduction

When thinking about a large language model input and output, a text prompt (sometimes accompanied by other modalities such as image prompts) is the input the model uses to predict a specific output. You don't need to be a data scientist or a machine learning engineer – everyone can write a prompt. However, crafting the most effective prompt can be complicated. Many aspects of your prompt affect its efficacy: the model you use, the model's training data, the model configurations, your word-choice, style and tone, structure, and context all matter. Therefore, prompt engineering is an iterative process. Inadequate prompts can lead to ambiguous, inaccurate responses, and can hinder the model's ability to provide meaningful output.
<table border="1">
<tr>
<td><b>Name</b></td>
<td colspan="3">1_1_movie_classification</td>
</tr>
<tr>
<td><b>Goal</b></td>
<td colspan="3">Classify movie reviews as positive, neutral or negative.</td>
</tr>
<tr>
<td><b>Model</b></td>
<td colspan="3">gemini-pro</td>
</tr>
<tr>
<td><b>Temperature</b></td>
<td>0.1</td>
<td><b>Token Limit</b></td>
<td>5</td>
</tr>
<tr>
<td><b>Top-K</b></td>
<td>N/A</td>
<td><b>Top-P</b></td>
<td>1</td>
</tr>
<tr>
<td><b>Prompt</b></td>
<td colspan="3">
                Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.<br/>
                Review: "Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. I wish there were more movies like this masterpiece.<br/>
                Sentiment:
            </td>
</tr>
<tr>
<td><b>Output</b></td>
<td colspan="3">POSITIVE</td>
</tr>
</table>

Table 1. An example of zero-shot prompting

When zero-shot doesn't work, you can provide demonstrations or examples in the prompt, which leads to "one-shot" and "few-shot" prompting. General prompting / zero shot

## One-shot & few-shot

When creating prompts for AI models, it is helpful to provide examples. These examples can help the model understand what you are asking for. Examples are especially useful when you want to steer the model to a certain output structure or pattern.

A **one-shot prompt**, provides a *single* example, hence the name one-shot. The idea is the model has an example it can imitate to best complete the task.

A **few-shot prompt** <sup>7</sup> provides *multiple* examples to the model. This approach shows the model a pattern that it needs to follow. The idea is similar to one-shot, but multiple examples of the desired pattern increases the chance the model follows the pattern.The number of examples you need for few-shot prompting depends on a few factors, including the complexity of the task, the quality of the examples, and the capabilities of the generative AI (gen AI) model you are using. As a general rule of thumb, you should use at least three to five examples for few-shot prompting. However, you may need to use more examples for more complex tasks, or you may need to use fewer due to the input length limitation of your model.

Table 2 shows a few-shot prompt example, let's use the same gemini-pro model configuration settings as before, other than increasing the token limit to accommodate the need for a longer response.

<table border="1">
<tr>
<td><b>Goal</b></td>
<td colspan="3">Parse pizza orders to JSON</td>
</tr>
<tr>
<td><b>Model</b></td>
<td colspan="3">gemini-pro</td>
</tr>
<tr>
<td><b>Temperature</b></td>
<td>0.1</td>
<td><b>Token Limit</b></td>
<td>250</td>
</tr>
<tr>
<td><b>Top-K</b></td>
<td>N/A</td>
<td><b>Top-P</b></td>
<td>1</td>
</tr>
<tr>
<td><b>Prompt</b></td>
<td colspan="3">
<p>Parse a customer's pizza order into valid JSON:</p>
<p>EXAMPLE:</p>
<p>I want a small pizza with cheese, tomato sauce, and pepperoni.</p>
<p>JSON Response:</p>
<pre>...
{
  "size": "small",
  "type": "normal",
  "ingredients": ["cheese", "tomato sauce", "peperoni"]
}</pre>
<p>...</p>
<p style="text-align: right;"><b>Continues next page...</b></p>
</td>
</tr>
</table>