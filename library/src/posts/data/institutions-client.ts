import { institutions } from "@/lib/api/resources";

export type InstitutionDTO = institutions.InstitutionDetail;

export async function loadInstitutionClient(
  institutionName: string
): Promise<InstitutionDTO | null> {
  return institutions.getInstitutionByName(institutionName);
}
