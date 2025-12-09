import json

filters = {"displayName": {"$eq": "daveey.4x4.cvc.dr_v1_rem"}}
filters_str = json.dumps(filters)

query = """
query GetRunMetrics($entity: String!, $project: String!, $filters: JSONString!) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, first: 1) {
      edges {
        node {
          id
          name
          displayName
          state
          createdAt
          summaryMetrics
          config
          historyKeys
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

variables = {"entity": "metta-research", "project": "metta", "filters": filters_str}

print("Filters JSON:", filters_str)
print("\nVariables:", json.dumps(variables, indent=2))
