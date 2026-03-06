#!/usr/bin/env node
/**
 * RAG Query Script for MDN Documentation
 *
 * Complete RAG pipeline:
 * 1. Retrieve - search for relevant doc chunks
 * 2. Augment - Build context from retrieved chunks
 * 3. Generate - Use Claude to generate answer with context
 *
 * Supports both streaming and non-streaming modes.
 * This script serves as a reference for UI implementation.
 *
 * Usage:
 *   npm run ask "How do closures work in JavaScript?"
 *   npm run ask "What is async/await?" -- --stream
 *   npm run ask "Explain promises" -- --limit=10
 */

import "dotenv/config";
import { anthropic } from "@ai-sdk/anthropic";
import { generateText, streamText } from "ai";
import {
  generateQuestionEmbedding,
  searchSimilarChunks,
  type SearchResult,
} from "./search";
import postgres from "postgres";

// Initialize database connection
const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://example:example@localhost:5455/example";
const client = postgres(connectionString);

/**
 * Build context string from search results for the LLM prompt
 */
function buildContext(results: SearchResult[]): string {
  if (results.length === 0) {
    return "No relevant documentation found.";
  }

  const contextParts = results.map((result, index) => {
    const parts = [
      `--- Document ${index + 1} ---`,
      `Title: ${result.title || "N/A"}`,
      `Source: ${result.source}`,
    ];

    if (result.heading) {
      parts.push(`Section: ${result.heading}`);
    }

    if (result.slug) {
      parts.push(
        `URL: https://developer.mozilla.org/en-US/docs/${result.slug}`,
      );
    }

    parts.push(
      `Relevance: ${(result.similarity * 100).toFixed(1)}%`,
      `\nContent:\n${result.text}`,
      "", // Empty line for spacing
    );

    return parts.join("\n");
  });

  return contextParts.join("\n");
}

/**
 * Generate a system prompt for the MDN documentation assistant
 */
function getSystemPrompt(): string {
  return `You are an expert JavaScript/Web Development assistant with deep knowledge of MDN (Mozilla Developer Network) documentation.

Your role is to:
- Answer questions accurately based on the provided MDN documentation context
- Explain concepts clearly with examples when helpful
- Reference the specific MDN pages when relevant
- Admit when the provided context doesn't contain enough information to fully answer
- Use proper technical terminology but explain it in an accessible way

When answering:
- Prioritize information from the provided context
- Include code examples from the context when available
- Mention which specific MDN page(s) the information comes from
- If the context is insufficient, say so clearly

Keep responses concise but thorough.`;
}

/**
 * RAG Query - Non-streaming version (for testing/CLI)
 */
async function ragQueryNonStreaming(
  question: string,
  options: {
    limit?: number;
    model?: string;
    temperature?: number;
  } = {},
): Promise<string> {
  const {
    limit = 5,
    model = "claude-3-haiku-20240307",
    temperature = 0.3,
  } = options;

  console.log("🚀 Starting RAG query...\n");
  console.log(`📝 Question: "${question}"`);
  console.log(`🤖 Model: ${model}`);
  console.log(`📚 Retrieving ${limit} relevant chunks...\n`);

  // Step 1: Generate embedding and retrieve relevant chunks
  const questionEmbedding = await generateQuestionEmbedding(question);
  const searchResults = await searchSimilarChunks(questionEmbedding, limit);

  if (searchResults.length === 0) {
    console.log("⚠️  No relevant documentation found.\n");
    return "I couldn't find any relevant documentation to answer your question. Please try rephrasing or asking about a different topic.";
  }

  console.log(`✅ Retrieved ${searchResults.length} relevant chunks\n`);
  console.log("📊 Top Results:");
  searchResults.forEach((result, index) => {
    console.log(
      `   ${index + 1}. ${result.title || result.source} (${(result.similarity * 100).toFixed(1)}%)`,
    );
  });

  // Step 2: Build context from retrieved chunks
  const context = buildContext(searchResults);

  // Step 3: Generate response using Claude
  console.log("\n🤖 Generating response with Claude...\n");

  const { text } = await generateText({
    model: anthropic(model),
    temperature,
    system: getSystemPrompt(),
    messages: [
      {
        role: "user",
        content: `Context from MDN Documentation:

${context}

---

Question: ${question}

Please answer based on the provided context.`,
      },
    ],
  });

  return text;
}

/**
 * RAG Query - Streaming version (for UI implementation)
 */
async function ragQueryStreaming(
  question: string,
  options: {
    limit?: number;
    model?: string;
    temperature?: number;
  } = {},
): Promise<void> {
  const {
    limit = 5,
    model = "claude-3-haiku-20240307",
    temperature = 0.3,
  } = options;

  console.log("🚀 Starting RAG query (streaming)...\n");
  console.log(`📝 Question: "${question}"`);
  console.log(`🤖 Model: ${model}`);
  console.log(`📚 Retrieving ${limit} relevant chunks...\n`);

  // Step 1: Generate embedding and retrieve relevant chunks
  const questionEmbedding = await generateQuestionEmbedding(question);
  const searchResults = await searchSimilarChunks(questionEmbedding, limit);

  if (searchResults.length === 0) {
    console.log("⚠️  No relevant documentation found.\n");
    return;
  }

  console.log(`✅ Retrieved ${searchResults.length} relevant chunks\n`);
  console.log("📊 Top Results:");
  searchResults.forEach((result, index) => {
    console.log(
      `   ${index + 1}. ${result.title || result.source} (${(result.similarity * 100).toFixed(1)}%)`,
    );
  });

  // Step 2: Build context from retrieved chunks
  const context = buildContext(searchResults);

  // Step 3: Stream response using Claude
  console.log("\n🤖 Streaming response from Claude...\n");
  console.log("=".repeat(80));

  const result = await streamText({
    model: anthropic(model),
    temperature,
    system: getSystemPrompt(),
    messages: [
      {
        role: "user",
        content: `Context from MDN Documentation:

${context}

---

Question: ${question}

Please answer based on the provided context.`,
      },
    ],
  });

  // Stream the response to console
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  console.log("\n" + "=".repeat(80) + "\n");
}

/**
 * Main function - handle CLI arguments
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Validate environment variables
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("❌ ANTHROPIC_API_KEY environment variable is required");
    console.log("\nAdd this to your .env file:");
    console.log("ANTHROPIC_API_KEY=your_api_key_here\n");
    process.exit(1);
  }

  if (!process.env.VOYAGEAI_API_KEY && !process.env.VOYAGE_API_KEY) {
    console.error("❌ VOYAGE_API_KEY environment variable is required");
    process.exit(1);
  }

  // Parse arguments
  let limit: number = 5;
  let streaming: boolean = false;
  let model: string = "claude-3-haiku-20240307";

  if (args.length === 0) {
    console.error("❌ No question provided");
    console.log("\nUsage:");
    console.log('  npm run ask "your question here"');
    console.log('  npm run ask "your question" -- --stream');
    console.log('  npm run ask "your question" -- --limit=10');
    console.log(
      '  npm run ask "your question" -- --model=claude-3-haiku-20240307',
    );
    process.exit(1);
  }

  // Extract question and flags
  const question = args.filter((arg) => !arg.startsWith("--")).join(" ");

  // Parse optional flags
  const limitArg = args.find((arg) => arg.startsWith("--limit="));
  if (limitArg) {
    limit = parseInt(limitArg.split("=")[1]) || 5;
  }

  const modelArg = args.find((arg) => arg.startsWith("--model="));
  if (modelArg) {
    model = modelArg.split("=")[1];
  }

  streaming = args.includes("--stream");

  try {
    if (streaming) {
      // Streaming mode (better for UI)
      await ragQueryStreaming(question, { limit, model });
    } else {
      // Non-streaming mode (easier to read in CLI)
      const answer = await ragQueryNonStreaming(question, { limit, model });

      console.log("=".repeat(80));
      console.log("💬 Answer:\n");
      console.log(answer);
      console.log("\n" + "=".repeat(80) + "\n");
    }
  } catch (error) {
    console.error("\n❌ Error:", error);
    process.exit(1);
  } finally {
    await client.end();
  }
}

// Run the script
if (require.main === module) {
  main().catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
}

// Export for use in UI/API routes
export {
  ragQueryNonStreaming,
  ragQueryStreaming,
  buildContext,
  getSystemPrompt,
};
