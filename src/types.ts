export interface DatasetMetadata {
  id: string;
  name: string;
  industry: string;
  description: string;
  rowCount: number;
  featureCount: number;
  features: string[];
  churnRate: number;
  correlationData: Array<{ feature: string; correlation: number }>;
  demographics: {
    labels: string[];
    churned: number[];
    retained: number[];
    title: string;
  };
  tenureDistribution: {
    bins: string[];
    churned: number[];
    retained: number[];
  };
}

export type ModelType = 'logistic_regression' | 'random_forest' | 'decision_tree' | 'gradient_boosting';

export interface Hyperparameters {
  learningRate?: number;
  numEstimators?: number;
  maxDepth?: number;
  regularization?: 'none' | 'l1' | 'l2';
}

export interface TrainingMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: {
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
    truePositive: number;
  };
  featureImportance: Array<{ feature: string; importance: number }>;
  lossCurve: Array<{ epoch: number; loss: number; valLoss: number }>;
  rocCurve: Array<{ fpr: number; tpr: number }>;
}

export interface CustomerInput {
  tenure: number;
  contract: 'month-to-month' | 'one-year' | 'two-year';
  monthlyCharges: number;
  techSupport: 'yes' | 'no' | 'no_internet';
  paymentMethod: 'electronic_check' | 'mailed_check' | 'bank_transfer' | 'credit_card';
  paperlessBilling: 'yes' | 'no';
  internetService: 'dsl' | 'fiber_optic' | 'no';
  multipleLines: 'yes' | 'no' | 'no_phone';
  partner: 'yes' | 'no';
  dependents: 'yes' | 'no';
  seniorCitizen: 'yes' | 'no';
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  id: string;
  timestamp: Date;
}
