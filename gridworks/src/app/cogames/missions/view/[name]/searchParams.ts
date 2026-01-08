import { createLoader, parseAsArrayOf, parseAsString } from "nuqs/server";

export const missionSearchParams = {
  variants: parseAsArrayOf(parseAsString, ","),
};

export const loadMissionSearchParams = createLoader(missionSearchParams);
