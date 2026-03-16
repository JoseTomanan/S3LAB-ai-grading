import type { LayoutLoad } from './$types.ts';
import type { TestInstance, TestItem } from '$lib/index.ts';


export const load: LayoutLoad = async ({ fetch, params }) => {
  const { test_id } = params;

  let test_instance: TestInstance | undefined = undefined;
  let test_items: TestItem[] | undefined = undefined;

  try {
    const response = await fetch(`/api/test_instances/${test_id}`);
    if (response.ok) {
      const result = await response.json();
      test_instance = {
        test_id,
        name: result.name,
        section_id: result.section_id,
        date: result.date,
        is_done_rendering: result.is_done_rendering
      };
    }
  } catch (e) {
    console.log("Failed to fetch test instance details:\n"+e);
  }

  try {
    const response = await fetch(`/api/test_items/${test_id}/items`);
    if (response.ok) {
      const result = await response.json();
      test_items = result.items;
    }
  } catch (e) {
    console.log("Failed to fetch test items:\n"+e);
  }

  // console.log(test_instance);
  // console.log(test_items);

  return {
    test_id,
    test_instance: test_instance,
    test_items,
  };
};
