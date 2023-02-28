// See https://aka.ms/new-console-template for more information
using OpenAI_API;
using QANDA.Net;

var api = new OpenAIAPI("your token");
var rag = new RAG(api);
var corpus = File.ReadAllLines("corpus.txt");
await rag.BuildCorpus(corpus);
var prompt = await rag.CreateSourceEmbedding("Who is John, Is he real", 5);
var answer = await rag.GenerateOutput(prompt);
Console.WriteLine(answer);

/*
output
load embedding form cache
FACT: John is a philanthropist and uses his fame to help a variety of charitable causes. He has also been recognized for his work in the field of virtual reality, having been awarded several awards for his work in the industry.
FACT: John's career began with a minor role in a popular sci-fi TV series. Soon, he was cast in a starring role in a virtual reality film. He was a hit with critics and audiences alike and his career quickly took off. He has since gone on to star in several feature films, as well as several video games and commercials.
FACT: John Smith is a virtual actor. He was created using advanced artificial intelligence and motion capture technology to bring his performance to life.
FACT: John loves to explore the world of virtual reality and is always looking for new ways to push the boundaries of technology and performance. He is a frequent speaker at industry conferences and is constantly pushing the boundaries of what is possible with virtual reality.
FACT: John is always looking for new projects and opportunities. He believes that virtual reality will continue to be an important part of our lives and strives to be at the forefront of its evolution.
QUESTION: Who is John, Is he real

ANSWER: John Smith is a virtual actor created using advanced artificial intelligence and motion capture technology. He is not a real person, but he is considered a pioneer in the field of virtual reality performance.
*/
