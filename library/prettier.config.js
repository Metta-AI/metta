module.exports = {
  trailingComma: "es5",
  plugins: ["prettier-plugin-tailwindcss"],
  overrides: [
    {
      files: ["tsconfig.json"],
      options: {
        parser: "jsonc",
      },
    },
  ],
};
