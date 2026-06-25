import { DatasetMetadata } from '../types';

export const telecomDataset: DatasetMetadata = {
  id: 'telecom',
  name: 'Telecom Customer Churn',
  industry: 'Telecommunications',
  description: 'Modeled on standard IBM Telco Customer Churn datasets. It captures demographics, subscription configurations (e.g. Fiber Optic, Tech Support, Streaming services), tenure, billing styles, and churn labels for 7,043 clients.',
  rowCount: 7043,
  featureCount: 20,
  features: [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 
    'TechSupport', 'OnlineSecurity', 'PaperlessBilling', 'PaymentMethod', 
    'Dependents', 'Partner', 'SeniorCitizen', 'MultipleLines', 'DeviceProtection'
  ],
  churnRate: 26.5,
  correlationData: [
    { feature: 'Month-to-Month Contract', correlation: 0.41 },
    { feature: 'Fiber Optic Internet', correlation: 0.31 },
    { feature: 'Electronic Check Payment', correlation: 0.30 },
    { feature: 'Monthly Charges', correlation: 0.19 },
    { feature: 'Paperless Billing', correlation: 0.19 },
    { feature: 'Senior Citizen Status', correlation: 0.15 },
    { feature: 'Partner / Dependents', correlation: -0.16 },
    { feature: 'Online Security (No)', correlation: 0.17 },
    { feature: 'Tech Support (Yes)', correlation: -0.16 },
    { feature: 'Tenure (Months)', correlation: -0.35 }
  ],
  demographics: {
    title: 'Churn Segmented by Contract Type',
    labels: ['Month-to-Month', 'One Year', 'Two Year'],
    churned: [1655, 166, 48],
    retained: [2220, 1307, 1647]
  },
  tenureDistribution: {
    bins: ['0-12m', '13-24m', '25-36m', '37-48m', '49-60m', '61-72m'],
    churned: [1037, 294, 180, 145, 120, 93],
    retained: [1147, 730, 652, 617, 712, 1316]
  }
};

export const fintechDataset: DatasetMetadata = {
  id: 'fintech',
  name: 'Fintech Credit Card Churn',
  industry: 'Finbanking & Fintech',
  description: 'Derived from bank customer portfolios tracking credit scores, consumer demographics (age, gender), active portfolio status, total balance, credit card possession, and estimated salaries across 10,000 retail accounts.',
  rowCount: 10000,
  featureCount: 13,
  features: [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender'
  ],
  churnRate: 20.3,
  correlationData: [
    { feature: 'Customer Age', correlation: 0.28 },
    { feature: 'Account Balance', correlation: 0.12 },
    { feature: 'Estimated Salary', correlation: 0.01 },
    { feature: 'Has Credit Card', correlation: -0.01 },
    { feature: 'Credit Score', correlation: -0.03 },
    { feature: 'Tenure', correlation: -0.01 },
    { feature: 'Number of Products (3+)', correlation: 0.16 },
    { feature: 'Inactive Member Status', correlation: 0.15 },
    { feature: 'Is Active Member (Yes)', correlation: -0.16 }
  ],
  demographics: {
    title: 'Churn Segmented by Customer Age Bin',
    labels: ['18-30 yrs', '31-45 yrs', '46-60 yrs', '61+ yrs'],
    churned: [142, 688, 1012, 195],
    retained: [1520, 4850, 1230, 363]
  },
  tenureDistribution: {
    bins: ['0-2 yrs', '3-4 yrs', '5-6 yrs', '7-8 yrs', '9-10 yrs'],
    churned: [412, 408, 403, 398, 416],
    retained: [1598, 1612, 1605, 1594, 1554]
  }
};

export const saasDataset: DatasetMetadata = {
  id: 'saas',
  name: 'SaaS Enterprise Churn',
  industry: 'B2B Software as a Service',
  description: 'Simulates product usage telemetry from enterprise subscription software, measuring monthly active team logins, open customer service tickets, total contract value, license utilization rates, and monthly usage trends.',
  rowCount: 5230,
  featureCount: 16,
  features: [
    'MonthlyLogins', 'SupportTickets', 'LicenseUtilization', 'ContractValue', 
    'TeamSize', 'SubscriptionType', 'UsageGrowthRate', 'FeatureAdoptionRate'
  ],
  churnRate: 14.8,
  correlationData: [
    { feature: 'Customer Support Tickets', correlation: 0.45 },
    { feature: 'Low License Utilization', correlation: 0.35 },
    { feature: 'Month-to-Month Billing', correlation: 0.22 },
    { feature: 'Slowing Usage Growth', correlation: 0.18 },
    { feature: 'Annual Revenue (ARR)', correlation: -0.05 },
    { feature: 'Contract Length (3yr)', correlation: -0.19 },
    { feature: 'High Feature Adoption', correlation: -0.24 },
    { feature: 'Monthly Logins / User', correlation: -0.38 }
  ],
  demographics: {
    title: 'Churn Segmented by Support Ticket Frequency',
    labels: ['0-1 ticket/mo', '2-3 tickets/mo', '4-5 tickets/mo', '6+ tickets/mo'],
    churned: [32, 145, 310, 287],
    retained: [2314, 1512, 531, 99]
  },
  tenureDistribution: {
    bins: ['0-6m', '7-12m', '13-18m', '19-24m', '25-36m', '37m+'],
    churned: [285, 194, 120, 85, 62, 28],
    retained: [520, 680, 714, 822, 943, 777]
  }
};

export const datasets: DatasetMetadata[] = [telecomDataset, fintechDataset, saasDataset];
