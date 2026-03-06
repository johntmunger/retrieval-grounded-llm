#!/usr/bin/env node
/**
 * Search Script for MDN RAG Application
 *
 * Performs vector similarity search using Voyage AI embeddings and PostgreSQL pgvector.
 * Uses cosine distance to find semantically similar document chunks.
 *
 * Usage: npm run search "your question here"
 */

import "dotenv/config";
import { createVoyage } from "voyage-ai-provider";
import { embedMany } from "ai";
import postgres from "postgres";

// Initialize database connection
const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://example:example@localhost:5455/example";
const client = postgres(connectionString);

// Initialize Voyage AI provider
const apiKey = (
  process.env.VOYAGEAI_API_KEY || process.env.VOYAGE_API_KEY
)?.trim();
if (!apiKey) {
  console.error("❌ No Voyage API key found");
  process.exit(1);
}

const voyage = createVoyage({ apiKey });
const embeddingModel = voyage.textEmbeddingModel("voyage-code-3");

/**
 * Search result interface matching our document_embeddings schema
 */
export interface SearchResult {
  id: string;
  text: string;
  source: string;
  heading: string | null;
  title: string | null;
  slug: string | null;
  similarity: number; // Cosine similarity score (0-1, higher is better)
}

/**
 * Generate embedding for a question using Voyage AI
 */
async function generateQuestionEmbedding(question: string): Promise<number[]> {
  console.log(`🔮 Generating embedding for question...`);

  try {
    const { embeddings } = await embedMany({
      model: embeddingModel,
      values: [question],
      providerOptions: {
        voyage: { inputType: "query", outputDimension: 1024 },
      },
    });

    console.log(
      `✅ Generated embedding with ${embeddings[0].length} dimensions\n`,
    );
    return embeddings[0];
  } catch (error) {
    console.error("❌ Error generating embedding:", error);
    throw error;
  }
}

/**
 * Search for semantically similar chunks using vector cosine similarity
 *
 * Uses PostgreSQL pgvector's cosine distance operator (<=>)
 * Similarity = 1 - cosine_distance (higher score = more similar)
 */
async function searchSimilarChunks(
  questionEmbedding: number[],
  limit: number = 5,
  similarityThreshold: number = 0.0,
): Promise<SearchResult[]> {
  console.log(
    `🔍 Searching for ${limit} most similar chunks using vector cosine similarity...`,
  );

  try {
    // Use cosine similarity for vector search
    // 1 - cosine_distance gives us cosine similarity (higher = more similar)
    const results = await client`
      SELECT 
        id,
        text,
        source,
        heading,
        title,
        slug,
        (1 - (embedding <=> ${JSON.stringify(questionEmbedding)}::vector)) as similarity
      FROM document_embeddings
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> ${JSON.stringify(questionEmbedding)}::vector
      LIMIT ${limit}
    `;

    // Filter by similarity threshold
    const filteredResults = results.filter(
      (result) => result.similarity >= similarityThreshold,
    );

    console.log(
      `✅ Found ${filteredResults.length} results above threshold ${similarityThreshold}\n`,
    );

    return filteredResults as SearchResult[];
  } catch (error) {
    console.error("❌ Error searching similar chunks:", error);
    throw error;
  }
}

/**
 * Format and display search results
 */
function displayResults(results: SearchResult[], question: string): void {
  console.log("\n" + "=".repeat(80));
  console.log(`📊 SEARCH RESULTS FOR: "${question}"`);
  console.log("=".repeat(80));

  if (results.length === 0) {
    console.log("🔍 No relevant chunks found above the similarity threshold.");
    console.log("💡 Try:");
    console.log("   - Rephrasing your question");
    console.log("   - Using different keywords");
    console.log("   - Lowering the similarity threshold");
    return;
  }

  results.forEach((result, index) => {
    console.log(`\n📄 RESULT ${index + 1}:`);
    console.log(`   📋 Title: ${result.title || "N/A"}`);
    console.log(`   📁 Source: ${result.source}`);
    console.log(`   🔗 Slug: ${result.slug || "N/A"}`);
    console.log(`   🎯 Similarity: ${(result.similarity * 100).toFixed(2)}%`);

    if (result.heading) {
      console.log(`   🏷️  Heading: ${result.heading}`);
    }

    console.log(`   💬 Content Preview:`);
    console.log(
      `   "${result.text.substring(0, 200).replace(/\n/g, " ")}${
        result.text.length > 200 ? "..." : ""
      }"`,
    );
    console.log(`   🆔 Chunk ID: ${result.id}`);

    if (index < results.length - 1) {
      console.log("\n" + "-".repeat(40));
    }
  });

  console.log("\n" + "=".repeat(80));
}

/**
 * Main function to perform search
 */
async function performSemanticSearch(
  question: string,
  limit: number = 5,
): Promise<SearchResult[]> {
  const similarityThreshold = 0.0; // Minimum similarity score (0-1)
  console.log("🚀 Starting search...\n");

  // Validate question
  if (!question || question.trim().length === 0) {
    throw new Error("Question cannot be empty");
  }

  // Validate environment variable
  if (!process.env.VOYAGEAI_API_KEY && !process.env.VOYAGE_API_KEY) {
    console.error(
      "❌ VOYAGE_API_KEY or VOYAGEAI_API_KEY environment variable is required but not set",
    );
    process.exit(1);
  }

  try {
    // Generate embedding for the question
    const questionEmbedding = await generateQuestionEmbedding(question);

    // Search for similar chunks using vector similarity
    const results = await searchSimilarChunks(
      questionEmbedding,
      limit,
      similarityThreshold,
    );

    // Display results
    displayResults(results, question);

    return results;
  } catch (error) {
    console.error("❌ Search failed:", error);
    throw error;
  }
}

/**
 * Handle command line arguments
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  let limit: number = 5;

  // Parse command line arguments
  if (args.length === 0) {
    console.error("❌ No question provided");
    console.log("Usage:");
    console.log('  npm run search "your question here"');
    console.log('  npm run search "your question" --limit=10');
    process.exit(1);
  }

  // Parse arguments: question --limit=N
  const question = args.filter((arg) => !arg.startsWith("--")).join(" ");

  const limitArg = args.find((arg) => arg.startsWith("--limit="));
  if (limitArg) {
    limit = parseInt(limitArg.split("=")[1]) || 5;
  }

  console.log(`Question: "${question}"`);
  console.log(`Limit: ${limit} results\n`);

  // Perform the search
  await performSemanticSearch(question, limit);
}

// Run the script
if (require.main === module) {
  main()
    .catch((error) => {
      console.error("Script failed:", error);
      process.exit(1);
    })
    .finally(() => {
      client.end();
    });
}

// Export for potential use as a module
export {
  performSemanticSearch,
  generateQuestionEmbedding,
  searchSimilarChunks,
};
