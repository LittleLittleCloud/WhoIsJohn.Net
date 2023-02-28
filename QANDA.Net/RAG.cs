using OpenAI_API;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace QANDA.Net
{
    internal class RAG
    {
        private Dictionary<string, float[]> embeddings;
        private OpenAIAPI openApi;
        private string cachePath = "RAG_cache.json";
        public RAG(OpenAIAPI openApi)
        {
            this.openApi = openApi;
            if(File.Exists(cachePath))
            {
                Console.WriteLine("load embedding form cache");
                var json = File.ReadAllText(cachePath);
                var cache = JsonSerializer.Deserialize<Dictionary<string, float[]>>(json);
                this.embeddings = cache;
            }
            else
            {
                this.embeddings = new Dictionary<string, float[]>();
            }
        }

        public async Task BuildCorpus(IEnumerable<string> corpus, bool byLine = true)
        {
            foreach(var corpusItem in corpus)
            {
                if (byLine)
                {
                    foreach(var line in corpusItem.Split(Environment.NewLine))
                    {
                        if (!embeddings.ContainsKey(line))
                        {
                            embeddings[line] = await openApi.Embeddings.GetEmbeddingsAsync(line);
                        }
                    }
                }
                else
                {
                    if (!embeddings.ContainsKey(corpusItem))
                    {
                        embeddings[corpusItem] = await openApi.Embeddings.GetEmbeddingsAsync(corpusItem);
                    }
                }
            }

            var json = JsonSerializer.Serialize(embeddings);
            File.WriteAllText(cachePath, json);
        }

        public async Task<string> CreateSourceEmbedding(string question, int referenceNumber = 10)
        {
            float cosineSimilarity(float[] a, float[] b)
            {
                int N = 0;
                N = ((a.Count() < b.Count()) ? a.Count() : b.Count());
                double dot = 0.0d;
                double mag1 = 0.0d;
                double mag2 = 0.0d;
                for (int n = 0; n < N; n++)
                {
                    dot += a[n] * b[n];
                    mag1 += Math.Pow(a[n], 2);
                    mag2 += Math.Pow(b[n], 2);
                }

                return Convert.ToSingle(dot / (Math.Sqrt(mag1) * Math.Sqrt(mag2)));
            }

            var questionEmbedding = await openApi.Embeddings.GetEmbeddingsAsync(question);
            var cosineSimilaritys = embeddings.Select(kv => (kv.Key, cosineSimilarity(kv.Value, questionEmbedding)))
                .OrderByDescending(item => item.Item2)
                .Take(referenceNumber);

            var sb = new StringBuilder();
            foreach(var item in cosineSimilaritys)
            {
                sb.AppendLine($"FACT: {item.Key}");
            }

            sb.AppendLine($"QUESTION: {question}");

            Console.WriteLine(sb);
            return sb.ToString();
        }

        public async Task<string> GenerateOutput(string prompt)
        {
            var result = await openApi.Completions.CreateCompletionAsync(prompt, max_tokens: 512);
            return result.Completions.First().Text;
        }
    }
}
