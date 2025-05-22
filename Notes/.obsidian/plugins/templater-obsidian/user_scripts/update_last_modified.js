/**
 * update_last_modified.js
 *
 * A Templater user script that sets the `last_modified` frontmatter
 * field to today's date (YYYY_MM_DD) whenever it's run.
 */
module.exports = async (tp) => {
  const today = tp.date.now("YYYY_MM_DD");
  await tp.frontmatter.update({ last_modified: today });
};
