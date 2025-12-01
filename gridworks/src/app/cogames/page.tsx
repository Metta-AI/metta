import { StyledLink } from "@/components/StyledLink";
import { cogamesMissionsRoute } from "@/lib/routes";

export default function CogamesPage() {
  return (
    <div className="p-4">
      <StyledLink href={cogamesMissionsRoute()}>Missions</StyledLink>
    </div>
  );
}
