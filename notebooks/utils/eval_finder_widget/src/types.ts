export interface EvalMetadata {
  name: string;
  category: string;
  description?: string;
  prerequisites: string[];
  tags: string[];
  is_completed?: boolean;
  performance_score?: number;
}

export interface EvalNode {
  name: string;
  category: string;
  eval_metadata: EvalMetadata;
}

export interface EvalCategory {
  name: string;
  category: string;
  children: EvalNode[];
}