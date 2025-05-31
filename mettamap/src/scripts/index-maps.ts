import { makeMapIndex } from "@/server/makeMapIndex";

function main() {
  makeMapIndex().then(() => {
    console.log("Map index created");
  });
}

main();
