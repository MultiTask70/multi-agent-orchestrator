import {
  ConversationMessage,
  ParticipantRole,
} from "../types";
import { isToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";
import { Classifier, ClassifierResult } from "./classifier";

export interface LocalClassifierOptions {
  // The API URL for your local model
  apiUrl: string;

  // The name of the model to use for classification
  modelName: string;

  // Optional: Configuration for the inference process
  inferenceConfig?: {
    // Maximum number of tokens to generate in the response
    maxTokens?: number;

    // Controls randomness in output generation
    temperature?: number;

    // Controls diversity of output via nucleus sampling
    topP?: number;

    // Array of sequences that will stop the model from generating further tokens when encountered
    stopSequences?: string[];
  };
}

export class LocalClassifier extends Classifier {
  private apiUrl: string;
  private modelName: string;
  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  private tools = [
    {
      name: 'analyzePrompt',
      description: 'Analyze the user input and provide structured output',
      input_schema: {
        type: 'object',
        properties: {
          userinput: {
            type: 'string',
            description: 'The original user input',
          },
          selected_agent: {
            type: 'string',
            description: 'The name of the selected agent',
          },
          confidence: {
            type: 'number',
            description: 'Confidence level between 0 and 1',
          },
        },
        required: ['userinput', 'selected_agent', 'confidence'],
      },
    },
  ];

  constructor(options: LocalClassifierOptions) {
    super();

    if (!options.apiUrl || !options.modelName) {
      throw new Error("API URL and model name are required");
    }

    this.apiUrl = options.apiUrl;
    this.modelName = options.modelName;

    // Set default value for maxTokens if not provided
    const defaultMaxTokens = 1000;
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? defaultMaxTokens,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };
  }

  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    chatHistory: ConversationMessage[]
  ): Promise<ClassifierResult> {
    try {
      const response = await fetch(`${this.apiUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.modelName,
          prompt: inputText,
          max_tokens: this.inferenceConfig.maxTokens,
          temperature: this.inferenceConfig.temperature,
          top_p: this.inferenceConfig.topP,
          stop: this.inferenceConfig.stopSequences,
          tools: this.tools,
        }),
      });

      const result = await response.json();

      const toolUse = result.choices[0].message?.tool_use;

      if (!toolUse) {
        throw new Error("No tool use found in the response");
      }

      if (!isToolInput(toolUse.input)) {
        throw new Error("Tool input does not match expected structure");
      }

      // Create and return ClassifierResult
      const classifierResult: ClassifierResult = {
        selectedAgent: this.getAgentById(toolUse.input.selected_agent),
        confidence: parseFloat(toolUse.input.confidence),
      };
      return classifierResult;
    } catch (error) {
      Logger.logger.error("Error processing request:", error);
      // Instead of returning a default result, we'll throw the error
      throw error;
    }
  }
}
