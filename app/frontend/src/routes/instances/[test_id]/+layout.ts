import type { LayoutLoad } from './$types.ts';
import type { TestInstance, TestItem } from '$lib/index.ts';


export const load: LayoutLoad = async ({ fetch, params }) => {
  const { test_id } = params;

  const [instanceResponse, itemsResponse] = await Promise.all([
    fetch(`/api/test_instances/${test_id}`),
    fetch(`/api/test_items/${test_id}/items`)
  ]);

  let test_instance: TestInstance | undefined = undefined;
  let test_items: TestItem[] | undefined = undefined;

  if (instanceResponse.ok) {
    const result = await instanceResponse.json();
    test_instance = {
      test_id,
      name: result.name,
      section_id: result.section_id,
      date: result.date,
      is_done_rendering: result.is_done_rendering
    };
  }

  if (itemsResponse.ok) {
    const result = await itemsResponse.json();
    test_items = result.items;
  }

  return {
    test_id,
    test_instance,
    test_items,
  };
};
